# Polylogue R&D bundle — file 2 of 6

> Preamble below is the MASTER SYNTHESIS, identical across all 6 files.
> The unique payload of THIS file is the 15 subagent outputs after the PAYLOAD divider.
> Files 1-6 together hold every substantial subagent output + the brainstorm corpus, size-balanced.

=====================================================================
  PART 1 - MASTER SYNTHESIS (preamble, same in all 6 files)
=====================================================================

---
created: "2026-07-05"
purpose: "Integrated master synthesis of the ~92-agent Polylogue R&D run (5 waves): convergent design themes, confirmed bugs, per-subsystem implementation specs, GPT-pro prompt fan-out, and the critical path. Written under context pressure to survive compaction."
status: "active"
project: "polylogue"
---

# Polylogue R&D — Master Synthesis (2026-07-05)

## 0. What this is and where the detail lives

A five-wave parallel R&D run (~92 background agents) over the Polylogue archive, driven from a
GPT-pro webui design session (`chatgpt-export:6a49af33…`, auto-captured via the browser extension).

- **Wave 1 (8):** wide ideation — product/analytics/self-referential/ecosystem/content/trust/UX/moonshots.
- **Wave 2 (14):** deeper ideation + first specs — queries-as-objects, measure-catalog, self-capture loop,
  activation, incident, temporal, economics, search, ingestion, proactive, missing-units, triage, episode, meta-quality.
- **Wave 3 (16):** new dimensions + 4 red-teams — whole-product, API surface, distribution, docs, workflows,
  observability, config, releases, **red-team ×4** (vision / honesty / security / cost), a11y, MCP, perf, demos.
- **Wave 4 (12 impl-specs):** measure-registry, context-scheduler, notification-policy, query/finding-objects,
  ingest-fidelity, compaction-event, content-variant, triage-inbox, delegation, episode, recursive-safety, temporal.
- **Wave 5 (42 impl-specs):** everything not yet spec'd + the red-team fixes + missing-units + RL/eval + 2 synthesis lanes.

**Full detail per agent** is in `/tmp/claude-1000/-realm-project-polylogue/<session>/tasks/*.output` (one JSONL per agent —
read a single file if you need one spec verbatim; do NOT bulk-read).
**Idea corpus (~170 deduped):** `.agent/scratch/rnd-brainstorm-2026-07-05.md`.
**GPT-pro prompt files (11 of 66):** `/realm/inbox/tmp/`.
**Sinnix fix already committed** (`f6de855`, not pushed): `mcp-polylogue` launcher strips the cloud-lane
`POLYLOGUE_ARCHIVE_ROOT=/tmp/polylogue-archive` leak so the recall MCP hits the real archive.

---

## 0.1 Provenance & confidence (read before trusting a citation)

This doc has **two groundedness tiers**, because a context-compaction fired mid-run and replaced the full subagent
*source outputs* for waves 1–4 (and the earliest wave-5 specs) with a summary:

- **Source-grounded (high confidence):** the §3 wave-5 entries and the §2b "later wave-5" bugs — these were written
  with the subagents' full `<result>` bodies in context, so their file:line claims are first-hand.
- **Summary-derived (verify before acting):** the §1 themes, the §2 confirmed-bug list, and the §3 **wave-4** entries —
  these were distilled from the pre-compaction summary, *not* re-read from the source `tasks/*.output` files this
  session. The claims are faithful at the level of *what* the bug/mechanism is; the specific **file:line citations are
  unverified-since-compaction**. The 50 source docs survive on disk at `<session>/tasks/*.output` and are the authority.

Practical rule: this doc is the **index and the reasoning**; before you touch code on any specific `file:line`, open that
spec's task-output file and confirm the anchor. Nothing is lost — it's a re-read away — but don't treat a summary-tier
citation as verified.

## 1. The convergent design themes (load-bearing — every spec assumes these)

1. **Queries + findings as first-class stored objects.** `query:<hash>` ObjectRef keyed on the *lowered,
   macro-expanded* AST (so equivalent queries collapse, mirroring content-hash idempotency); `result_sets`
   snapshots in the **derived** index tier (Merkle root over sorted member keys, `hash(query_id ‖ corpus_version)`);
   a `query_edges` DAG populated from the set-algebra EXPLAIN nodes; **`AssertionKind.FINDING`** reusing the
   existing pathology candidate→`judge_assertion_candidate`→promote lifecycle *verbatim*; a `StandingQueryStage`
   in the DaemonConverger; **findings-as-tests** (a promoted finding with `value.expected` becomes a re-runnable
   invariant). Query versioning via the existing `supersedes_json`.

2. **The measure algebra.** A measure = `⟨reducer(column) over unit-frame × grouping × window × comparison × uncertainty⟩`
   — a cartesian product, not hand-written `analyze` modes. Each measure declares a construct-validity registry row
   (construct, formula, correct denominator/unit-frame, evidence tier, top-2 confounds, coverage precondition).
   Uncertainty layer: Wilson for proportions, bootstrap CI for skewed median/pXX, a coverage gate. `count` is the
   trivial measure so there is **one** aggregate path. **`SampleFrame.LOGICAL_SESSION` is the default frame** —
   this is what structurally kills the ~32% lineage cost double-count.

3. **The recursive-safety gate** (coherent subsystem, not a flag — matters *because* the browser extension
   auto-captures the operator's own R&D). Mechanisms: agent-authored assertions default `candidate` + `inject:false`;
   a **closed-loop laundering gate** (a claim whose evidence resolves *only* to other agent-authored sessions →
   quarantine until an external-grounded citation or human judgment breaks the cycle); the inject gate requires
   `author_kind='user'` (promotion via `judge_assertion_candidate` flips it); never inject/alert/mine on
   `material_origin ∈ {generated_context_pack, runtime_context, runtime_protocol, tool_result}`; provenance-cycle
   quarantine reusing `TopologyEdgeStatus.quarantined`; auto-downgrade findings on evidence-content-hash drift.

4. **Content-hash citation anchors.** Block identity today is `message_id:position` — position shifts on re-ingest
   and fork-replay. The fix (webui/CIF specs): add `blocks.block_content_hash` (derived, materializer-populated),
   resolve anchors against the *composed* lineage transcript by hash not offset, three states
   (`exact` / `drifted` / `deleted` / `quarantined`). This is the atom the cockpit, findings, and CIF all stand on.

5. **The episode unit.** A logical task spanning sessions/tools/time, a tier *above* within-provider lineage,
   edges thematic/temporal not identity-shared. 4-signal scorer (repo × repo-conditioned-time-kernel × embedding ×
   shared-artifact) with a **hard false-merge floor** (a strong structural corroborator — shared SHA / error
   fingerprint / same-repo+time — required before any `corroborated+` tier; thematic-only stays `candidate`).

6. **Corpus-compaction (`find <q> | compact`).** Token-budgeted, decision-dense digest for feeding an external LLM
   (the R&D-flywheel enabler): drop tool-spam via `material_origin`, keep human turns + decisions + error→fix +
   outcomes, dedup lineage prefixes, fit a context window, emit a fidelity drop-manifest.

7. **Construct-validity is executable everywhere.** Coverage-preconditions that *refuse rather than fabricate*;
   evidence-tier footnotes on every number; the `unknown` bucket never folded into a denominator; text-mined
   fields tagged `text_derived`/unverified in the payload contract, not just the bundle renderer.

8. **"Wire what exists" beats "build new."** The daemon already has a 5-backend notification fan-out (carries only
   ops alerts); `score_components` already carries the per-lane RRF decomposition; postmortem-bundle machinery,
   lease-safe blob GC, and the coordination envelope + degradation ladder are already shipped. Most wins are routing
   an existing pipe to a new signal, not new infrastructure.

9. **Durability-keyed schema discipline (the recurring "how").** *Almost nothing touches the durable tier.* New
   `AssertionKind` values are schema-free `TEXT` (no migration). Derived index/embeddings changes edit canonical DDL +
   bump version + rebuild (`ops reset --index && polylogued run`), **never an upgrade helper** (`devtools lab policy
   schema-versioning` rejects them). Durable source/user changes need a numbered additive migration + a **verified
   backup manifest** + one `PRAGMA user_version` step. Every new AssertionKind/enum is embedded in `render openapi` +
   `render cli-output-schemas`; every new module breaks the topology projection — regenerate and grep `render all
   --check` for `out of sync` (the tail line lies).

---

## 2. Confirmed bugs (red-team, file:line'd, actionable NOW)

### Cost/lineage (from the cost red-team + cost-correctness spec)
- **Day/week summaries sum PHYSICAL sessions** (`session_summaries.py:98-104`) → ~32% lineage double-count of
  cost/duration/messages/words. Fix: logical-grain rollups (one representative per `logical_session_id`).
- **Exact provider-cost path is DEAD** — `pricing.py:615` `_session_level_estimate` returns `None`;
  `session_reported_costs` is written but read by nothing → every subscription-cost figure is silently $0. Fix: wire the reader.
- **Missing credit rates for Opus 4.7/4.8** (`subscription_pricing.py:84-105` only has 4-5/4-6) →
  `get_credit_rate("claude-opus-4-8")=None` → credit_cost=0 for the most-used model. Fix: add rates + a test that
  `MODEL_CREDIT_RATES ⊇ curated Anthropic keys`; lock `output = 5×input`.
- **Subscription $/credit hardcodes Pro tier** (`cost_compute.py:133` `/21_700_000*20.0`). Fix: parametrize by
  `user_settings` subscription tier (assertion `subscription_tier`).
- **`cost_enrichment.py:54` downgrades EXACT→catalog.** Fix: strength lattice, never demote provider_reported.
- **`wall_duration_ms` summed across parallel subagents** (>86.4M ms/day representable). Fix: interval-union.
- **Codex disjoint-lane double-count** (the 7.69× class, fixed once in `3938bc6c2`, unguarded). Fix: assert
  `cached ⊆ input`, `reasoning ⊆ output` at parse.

### Honesty (from the honesty red-team + honesty-audit spec)
- **`_RIGOR_MATRIX` covers only 5 of 11 number-bearing products** — cost/coverage/tool/debt silently skipped
  (`audit.py:188` iterates contracts, not the registry). Fix: iterate the registry, emit `coverage_status=uncovered`.
- **`transforms.py` text-mines** commit SHAs / decisions / caveats / test-pass-counts from prose into forensic
  bundles while "no regex over prose" only holds for the exit-code axis. Fix: tag `text_derived` in the payload model.
- **`classify_aggregate_hwm_source` launders provenance to `provider_ts`** (`temporal_source.py:97`). Fix: propagate
  the *weakest* source.
- **Attachment referenced-vs-stored bytes** reported as retrievable (`#2468`; 967 acquired vs 6,414 unfetched,
  ~13.6 GB referenced-never-stored). Fix: split `referenced_bytes` from `stored_bytes` on every surface.

### Security (from the security red-team + security-hardening spec)
- **DNS-rebinding reads the whole archive**: no `Host` check on GET routes; `Origin` checked only on POST and skipped
  when absent (`daemon/http.py`). Fix: central `Host`/`Origin` allowlist middleware before dispatch.
- **Browser-capture receiver has no auth on loopback** → any local process POSTs forged captures into the spool.
  Fix: auto-mint a `0600` receiver token; `hmac.compare_digest`; restrict `?access_token=` to the SSE route; spool quota.
- **`reset.py` tombstones before the `--yes`/preview gate** (jnj.5). Fix: one shared mutation-audit contract with a
  dry-run preview across excise/reset/delete/MCP-admin.

### Live bugs (from the temporal spec)
- **`RELATIVE_BASE = datetime.now()` at import** (`core/dates.py:37`) → `since:7d` anchors to process-start,
  unpatchable by `frozen_clock`. Fix: route through a `core/clock.py` seam.
- **`sort_key_ms = COALESCE(...,0)`** vanishes timeless sessions from `since:` windows and pins them at 1970 in sorts.
  Fix: half-open `[since,until)`, explicit `IS NULL` handling, surfaced as `time_confidence=synthetic`.

---

### Live bugs found by the later wave-5 specs (not in §2 above — all file:line'd in their task outputs)
- **`ops doctor cleanup_orphans` can delete an in-flight leased blob** (blob-GC spec, risk R1). `run_blob_gc` is
  lease/ref/generation-safe, but `BlobStore.detect_orphans`/`cleanup_orphans` (the `ops doctor` path) compares disk
  against **only `raw_sessions.raw_id`** — ignores `pending_blob_refs`, `blob_refs`, and the generation-age gate. This
  is the real "#818 orphan-detection bug." Fix: make `cleanup_orphans` lease+generation-aware or hard-gate it behind
  `run_blob_gc`. Not optional.
- **`blackboard_post` lets an agent write `author_kind=agent` that lands `status=ACTIVE`** (annotation-recipe spec) —
  a recursive-safety hole: an agent's own claim can self-inject as authoritative. Fix: one `coerce_agent_authored`
  chokepoint inside `upsert_assertion` forcing all non-`user` authors → `CANDIDATE` + `inject:false` (never resurrecting
  a terminal-judged row). This is the load-bearing invariant recall/recipe/distillery/goal/decision all assume.
- **Identity collision *beneath* the origin-collapse aggregate bug** (provider→origin spec). Beyond the §1 aggregate
  double-count: `session_id = origin || ':' || native_id` means a `gemini-export` and a `drive-takeout` session sharing a
  `native_id` are **already one physical row** — undetectable, un-splittable even by reparse. The `source_family` column
  + `lossy_grouping` marker fixes *aggregation* honesty, not identity; file the identity half as separate durable scope.
- **`user_settings` table is dead** (config-engine spec): declared in `user.py` + migration `004`, but **zero read/write
  helpers exist** — unwired, empty. Unify into `assertions` as `AssertionKind.setting` (drop the table via copy-forward).
- **`session_phases` deliberately has no `kind`** (units-D grounding correction) — the intent-classified equivalent is
  `session_work_events`, which **already exists as durable index rows**. Re-adding `kind` reverts a construct decision;
  `phase-segment` is a DSL projection over work-events, not a new table.

## 3. Spec inventory (implementation-grade; each ships DDL + tier/regime + algorithms + tests + a 4–8 bead breakdown)

Format: **name** — tier/regime · key mechanism · headline risk.

### Wave-4 (all read in full)
- **measure-registry** — index/no-DDL (compute-on-read) · the 5-tuple algebra + 16 measures as registry rows +
  coverage-gate that *refuses* rather than fabricates + Wilson/bootstrap uncertainty · risk: any surface bypassing
  the single aggregate path escapes the gate. 8 beads.
- **query/finding-objects** — index v25 (query_defs/query_edges/result_sets/result_set_members) + user.db
  `FINDING` enum (no migration) · content-addressed query id, StandingQueryStage, findings-as-tests · risk:
  canonical-form drift orphans every result_set + finding evidence_ref. 6 beads. **Depends on set-algebra (fnm.13).**
- **recursive-safety** — user.db v5 (+`provenance_state` reusing TopologyEdgeStatus, +`safety_json`; additive migration) ·
  closed-loop laundering CTE over the evidence graph, `author_kind='user'` inject gate, drift auto-downgrade on the
  *durable* content-hash · risk: too-tight quarantines legit agent R&D (operator_command counts as grounding). 6 beads (37t.13-18).
- **context-scheduler** — user.db JSON-only (programmable `context_policy_json` windows) + ops.db ledger + read-access log ·
  ranking = staleness-decay × topic-proximity × attention; trust-class as slice-1-not-retrofit · risk: trust retrofit = injection hole. 8 beads (37t.11.a-h).
- **compaction-event** — source.db v3 (durable, `compaction_snapshots` + blob) + index `compaction_loss` · loss-forensics
  diff (retained/lost/transformed over file-path/tool-outcome/marked-decision/cited-ref), honest degradation on
  JSONL-boundary snapshots, loss-record survives the *next* compaction · risk: PreCompact payload may lack the assembled context. 7 beads (gjg.1-7).
- **episode** — index v25 (episodes/episode_members/episode_edges/produced_refs) + embeddings v2 (session_embeddings) +
  user.db `EPISODE_{CONFIRM,SPLIT,REJECT}` · 4-signal scorer + hard false-merge floor + anchor-keyed deterministic id ·
  risk: embedding cosine conflates thematic-similarity with same-task. 7 beads.
- **delegation** — index v25 (`delegations` VIEW + `primary_model_*` cols) · delegation-yield measure + delegation-card +
  the Fable `.polydemo`; `result_status` derives only from `actions.is_error`/exit_code (`unknown` never in the ROI denominator) ·
  risk: provider-vocab leak (model→family must project origin-vocab). 6 beads.
- **content-variant** — user.db v5 (content_variants/variant_nodes/variant_alignments; additive migration) + index
  `block_language_facts` · mechanical-vs-generative provenance axis, coverage>0 write invariant, `source_content_hash`
  staleness (never auto-repaint), dark-matter rendering · risk: false-stale storms from fingerprint drift; durable/derived
  orphaning during rebuild. 6 beads.
- **triage-inbox** — index v25 (`worth_reviewing_score` + breakdown + gate on session_profiles) + user.db `TRIAGED`
  (no migration) · time-invariant score materialized, inverted-U staleness applied at read, `WHERE NOT EXISTS triaged`
  empties the inbox, snooze-with-wake · risk: materialized staleness goes stale; gates hide instead of resolve. 6 beads.
- **ingest-fidelity** — source.db v3 (+`parser_fingerprint`, +`decode_failure_class`) + index `raw_fidelity` · byte-fidelity
  ratio (band per origin, not absolute), round-trip reconstruction (structural equality is the real bar), unparsed-key
  census, misclassification tripwire, `parser_fingerprint`-driven reprocess-on-improvement · risk: ratio misread as a bug. 8 beads (F1-F8).
- **temporal** — index-only + `core/clock.py` seam · four typed time-kinds, `sort_key_provenance`, skew-band **quantization**
  (not pairwise — transitivity), `[since,until)`, tz-unknown-by-default, backfill-never-moves-sort_key; fixes the two live
  bugs · risk: de-generating sort_key_ms reorders history/breaks cursors. 8 beads (cpf.*).
- **notification-policy** — user.db `NOTIFICATION_POLICY` (no migration) + ops.db ledger · routes CONTENT signals through
  the existing fan-out (`Notice` severity + `content` family), standing queries as producers, "you're repeating a past
  mistake" nudge, token-bucket fatigue control, no-self-alert · risk: fatigue defeats adoption → ship ledger-first. 8 beads (notify.1-8).

### Wave-5 (read in full so far)
- **blue-green index rebuild (b5l)** — durable pointer file + ops mirror · generation-suffixed `index.gN.db` + atomic
  pointer swap (<100ms) + delta-replay + lease-safe reaping; kills the 20-40min degraded window · risk: delta-replay race
  at swap boundary. 7 beads (b5l.1-7).
- **cross-machine sync** — sync only durable tiers · source.db content-hash union (idempotent+commutative), user.db
  assertion natural-key LWW merge, rebuild derived on peer, `.well-known/ai-sessions` manifest · risk: `raw_id` contains
  machine-local `source_path` → same session gets two ids across machines. 6 beads.
- **python-sdk** — `polylogue.sdk` + `polylogue.models` · ~20-method curated surface, sync facade (the async-only facade
  is why lynchpin bypasses via raw sqlite + reimplemented models + a stale `FROM conversations` query), frozen model
  re-exports, schema pin-and-warn, layering lint · risk: freezing origin vocab mid-retirement. 8 beads.
- **vec0-ANN** — embeddings v2 · int8-primary + f32-rerank vec0 (4× smaller scan), origin partition-key + centroid IVF
  prefilter, multi-probe batching to kill the 20-scan session-similarity fan-out, **recall@k eval from fork/resume
  lineage pairs (free positive labels)** · risk: quantization silently degrades recall at real scale. 6 beads (mhx.6.*).
- **mcp-collapse** — ~130 tools → **9 verbs** (`query`/`get`/`explain`/`context`/`correlate`/`coordinate`/`assert`/`retract`/`maintenance`) ·
  one `query(expression)` over the DSL absorbs ~40 read tools; resources/subscribe + `list_changed`; saved-views as
  dynamic prompts; `assert`/`retract` over the unified assertions table · risk: silent capability loss (per-tool
  equivalence goldens before deletion). 8 beads.
- **second-brain graph** — index v25 (entities/entity_mentions/entity_topics + entity_backlinks VIEW) · structural-vs-
  candidate mention split (recursive-safety-gated prose mining), bare-`#N` repo-scoped, topic co-occurrence clustering ·
  risk: prose-mining fabrication feedback loop (self-archiving). 6 beads.
- **session-replay** — ops.db (disposable) replay_runs/replay_steps · phase-1 DRY plan from the `actions` stream (no
  execution), phase-2 gated worktree re-execution with network-deny + frozen-clock + fail-closed classification ·
  risk: false anchors (only `explicit_ref`/`origin_reported` SHAs gate a checkout); silent step incompleteness. 8 beads (repro-1..8).
- **security-hardening** — runtime+config only, no DB migration · central Host/Origin gate, auto-minted receiver token,
  `hmac.compare_digest`, spool governor · risk: breaking the same-origin web shell (allowlist must admit its Host). 7 beads.
- **polish-fts** — index v25 (`unicode61 remove_diacritics 2` + a `pl_fold` write+query symmetry for the precomposed
  `ł`/`Ł` that `remove_diacritics` *cannot* fold) + `block_prose_lang` + trigram fallback lane · risk: query/index folding
  drift = silent recall loss. 6-8 beads.
- **attachment-bytes** — no schema change (columns already honest) · project `acquisition_status`+`blob_hash` through the
  4 read paths that drop them; two-sum accounting (`referenced` vs `stored`) everywhere + a lint forbidding bare byte
  totals; census (83u.6) + classifier (83u.4) + citation-anchor three-state resolver · risk: `size_bytes` compat alias
  stays the leak (prefer rename so mypy surfaces call sites). 6 beads.
- **own-export/CIF** — index additive + Origin enum · `polylogue-export` origin whose parser reconstructs the *embedded*
  origin so `import(export(A))` is a content-hash **no-op** (free correctness invariant + federation primitive);
  `content_hash_algo` pinned in the envelope · risk: hash-algorithm drift breaks federation. 6 beads.
- **cost-correctness** — index v25 + user.db tier assertion · the precedence lattice (provider_reported ≻ catalog_priced ≻
  heuristic, never downgrade EXACT), logical-grain rollups, per-model credit rates, tier-parametrized $/credit, disjoint-
  lane guard, wall interval-union · risk: logical-representative selection is owned by 4ts. 6 beads (f2qv.*).
- **honesty-audit** — index derived + pure contracts · field-level `RigorFieldContract` + `ProvenanceClass`, iterate the
  *registry* (emit `uncovered`), number-over-empty gate, weakest-source propagation, text_derived tags, referenced-vs-
  stored split, a `devtools lab policy insight-honesty` CI gate · risk: `nullable_when_ungrounded` breaks byte-compat
  consumers (deliberate; gate behind materializer-version bump). 7 beads.
- **zstd/tiering economics** — source v3 (blob_placement/blob_dicts/blob_tombstones/frozen_segments) · address stays
  `SHA(uncompressed)` forever; zstd per-origin trained dicts (36GB→4-7GB), access-temperature cold shard, embeddings
  int8, yearly-epoch freeze, single `blob-compact` walk, per-origin bytes/session SLO breaker · risk: any read/verify
  path that forgets to decompress silently breaks the backup verifier. 8 beads.
- **product-positioning** (plan) · keep "system of record for AI work" as the category anchor, lead with the
  **flight-recorder** hook + the **24.1% silent-proceed** hero finding (published, reproducible, construct-valid);
  honesty-benchmark is the *launch campaign* not the product identity; hold all memory-uplift claims capability-phrased
  until `cfk` reports (the `jxe` pilot closed *negative*).
- **secret-redaction/forget** — source v3 (`excision_tombstones`) + ops audit + user.db `SECRET_CANDIDATE` · scan-at-read
  overlay (never persists the value), judgment-queue, hash-tombstone excision, one mutation-audit contract, converger
  reconciliation invariant, **idempotency resurrection guard** (re-ingesting the original source can't resurrect) ·
  risk: parse-drift asymmetry vs the content-hash key → block-granularity + span-text-hash probe. 8 beads.
- **adoption/install** — ops.db v2 (hook_liveness/doctor_snapshots), live-computed adoption · `polylogue install`
  (idempotent all-hook wiring, match-by-command-substring so foreign hooks never clobbered) + `polylogue doctor`
  (liveness heartbeat + 5-way "why-zero-usage" diagnosis with a *relevance control*) + PreCompact recall + SessionStart
  brief · risk: adoption metric false-alarms → the signal gets ignored (the very failure it exists to prevent). 8 beads.
- **webui evidence-cockpit (bby.11)** — index `blocks.block_content_hash` (the re-ingest-stable anchor) + user.db
  `citation_anchor`/`evidence_report`/`report_basket` (no migration) · TS+Preact SPA hydrating a server-rendered
  Markdown/HTML twin (progressive enhancement), evidence-basket→report, integrity verifier, force-directed lineage graph ·
  risk: duplicated-block ambiguity (acompact replays a hash N×). 8 beads.
- **s7ae coordination** — user.db v5 (`coordination_message`/`coordination_ack` + a virtual `expires_at_ms` generated
  column) · scoped message bus on the assertions table, `query:<hash>` live-query refs (notepad→task bus), coordination
  as a `ContextSource` (37t.11), the two-agent same-repo proof harness · risk: hard dependency on unbuilt 37t.11 (the
  advisory leg is the injection surface). Envelope/degradation-ladder already shipped. 8 beads (s7ae.3/.5).
- **demo/proof engine** — additive · `.polydemo` executable format (frontmatter budget + product-primitive CLI steps +
  content-addressed `finding_id` = hash(claim+metric+anchor+sorted-refs+corpus-datasheet-hash)) + round-trip evidence-ref
  resolver gate + construct-validity pre-render tripwire + refusal manifest + demo-as-CI-test (finding_id drift breaks
  the build); reframes the 60KB `claim_vs_evidence.py` as the first `.polydemo` on the sru.1 DSL · risk: compositionality
  erosion (steps shelling to python recreate the monolith — parser accepts only `polylogue …` argv). 7 beads (B1-B7).

### Wave-5 — the remaining ~22 (ALL now complete; full detail in task files). Sharpest finding each surfaced:
- **provider→origin** — `source_family` column + data-driven `lossy_grouping` marker (fires only when a projected origin
  merges ≥2 families); identity-collision residual filed separately (see §2b). One projection chokepoint or surfaces drift.
- **FTS-coherence** — the `messages_fts.rowid == blocks.rowid == docsize.id` identity is the keystone; metamorphic law
  (any block op-sequence ⇒ 0 missing/excess) + O(1) ledger-only drift gauge; rowid-reuse must also check `block_id`.
- **search-relevance** — session resolution lives in **two** paths (`hybrid_sessions` + `archive_execution`); collapse +
  `variant_count` must land in one shared helper or lineage siblings duplicate (the #2470 class). `AssertionKind.RELEVANCE` LtR corpus, nDCG gate.
- **migration-fuzz** — the chain is **sparse** (source starts `002`, user `004`); fuzz `start_version` must map to real
  historical DDL or the contiguous-chain check legitimately raises. CHECK↔`Literal` drift tripwire via site-count assert.
- **config-engine** — 5-layer *deployment* resolver already exists (`config.py`) but has no `db` layer; add a
  scope(global→repo→origin→surface)×actor(operator→agent→harness)×override(flag>env>file>db>default) resolver over
  settings-as-assertions; learned-defaults self-exclude `config`/`judge` telemetry.
- **observability/SLOs** — one measure-sample stream (`slo_samples`, ops.db) + reducers (level/quantile/slope/eta/burn);
  the honesty keystone is the **idle-vs-stalled verdict** (backlog>0 is a defect only when not draining AND work offered);
  `ingest_latency` must scope to live-tail origins (bulk-export excluded by construction).
- **units A** — entity-mention=**TABLE** (regex extraction, structural-vs-candidate trust axis); world-effect=**VIEW**
  (cause from `observed_event.evidence_refs_json[0]`→`actions`); verification-run=**VIEW** (pass from exit_code keystone).
- **units B** — turn-pair=**VIEW** (material_origin adjacency); artifact=VIEW→table (distinct from `files`/`raw_artifacts`
  name-collision); **correction-edge MUST be a runtime query method, not a stored VIEW** — SQLite forbids a persistent
  VIEW over the ATTACHed `user_tier`. Correction anchors are mostly `session:`-coarse (limits error-rate-per-tool).
- **units C** — topic-cluster (HDBSCAN+kmeans fallback, content-addressed ids for timeline stability); **project is the
  only durable-tier unit** (user.db v5, gizmo_id extraction prereq, segment-aligned cwd match); cross-origin-thread
  (differ-origin ∩ ¬lineage ∩ hard-signal floor + hub-merge guard).
- **units D** — phase-segment projects over existing work-events (no new table); goal/decision are construct-gated
  candidates via the **existing** candidate→judge state machine; the recovery-digest incident is the shared regression test.
- **RL/eval (fs1.5)** — pure read projection to Atropos JSONL; **verify the live Atropos schema first** (round-trip via
  `jsonl2html.py`); corrections are **session-scoped** so reward-model reports AUC *with base-rate + n*; `PROMPT_EVAL` kind exists.
- **cross-project recall MCP** — text-hint→vector via existing `VectorProvider.query` (needs live Voyage key → honest FTS
  fallback); trust-class (OPERATOR/QUOTED/SYSTEM) is the laundering barrier, built in from slice 1, not retrofit.
- **spec-cards + TQI** — spec-card reproducibility gated on high-confidence commit attribution; TQI's `fragmentation_sub`
  is heuristic (phases aren't intent), lowest weight; ship strictly as reward-*shaping*, never sole reward (Goodhart).
- **model-drift observatory** — blocked on 9l5.7; intent-anchor validity is the weakest link (mandatory embed-coverage
  gate); a model upgrade re-keys the cohort, so changepoints are candidate + nearby-event, never causal.
- **read-access-log + attention** — ops.db is already multi-writer (per `daemon/events.py`); in-process debounce +
  decayed counter; **`context_inject` excluded from the attention signal** so the scheduler can't reinforce its own injections.
- **prompt-distillery** — recipes/prompts live in git-YAML (code under review), not user.db; distilled prompts are
  `PROMPT_TEMPLATE` candidates; A/B evaluator returns INSUFFICIENT_EVIDENCE below floor (never fabricates a win).
- **corpus-compaction** — `find <q> | compact` is a read-algebra point (Selection×CompactionProjection×COMPACT render);
  token proxy is word-count → apply a `0.72` BPE derate; outcomes from keystone fields only (anti-fabrication).
- **standing-query** — dedup by **stable finding-identity set**, not events → re-ingest storm is structurally impossible
  (same content-hash ⇒ same finding_id ⇒ already in membership); baseline-then-notify; self-trigger firewall excludes `notice.*`.
- **annotation-recipe** — the real completeness gap is **query-back**: assertions are MCP-`list`-only today, not in the
  `find` DSL; phase-1 is honest about that. Enforce recursive-safety at the single `upsert_assertion` chokepoint.

---

## 4. Cross-cutting implementation patterns (the recurring "how" — apply to every bead)

- **Prefer schema-free `AssertionKind` (TEXT, no CHECK)** for new user-tier vocabulary — no migration; but regenerate
  `render openapi` + `render cli-output-schemas` and add a `user_audit` surface entry, or discovery/audit tests fail.
- **Derived tier (index/embeddings) = edit canonical DDL + bump version + rebuild plan**, never an upgrade helper.
  Batch same-tier bumps from ready beads before triggering a live 38GB rebuild. **Blue-green (b5l) removes the downtime** —
  land it early and every subsequent index bump is a non-event.
- **Durable tier (source/user) = numbered additive migration + verified backup manifest + one `PRAGMA user_version` step.**
  The specs that need this: recursive-safety (user v5), content-variant (user v5), s7ae (user v5), compaction-event
  (source v3), ingest-fidelity (source v3), secret-redaction (source v3), zstd (source v3). **Consider batching the user
  v4→v5 and source v2→v3 bumps** so the live archive migrates once.
- **`LOGICAL_SESSION` grain is the default** for every measure/rollup/cost/analytics; `session_count` stays physical,
  `logical_session_count` is the distinct count. This is owned by `4ts` — gate cost-correctness B5 and episode cost
  rollup on it.
- **Every new module → regenerate topology projection** (`render topology-projection && topology-status`).
- **`ContextSource` protocol (37t.11) is the spine** for context-scheduler, notification advisories, coordination
  advisories, adoption SessionStart brief, and compaction re-grounding — build it first; its trust-class gate is a
  *security* invariant, not a nicety.
- **`frozen_clock` + `core/clock.py` seam** for all time-sensitive tests (the `verify-test-clock-hygiene` lint enforces it).

---

- **BATCH THE SCHEMA BUMPS — the single biggest operational insight across the specs.** Nearly every derived-tier spec
  bumps **index.db v24→v25** (units A/B/C/D, topic-cluster, entity-graph, spec-cards, cost-correctness, polish-FTS,
  query-objects, delegation, content-variant, honesty-audit, TQI, provider→origin source_family, drift…). Several durable
  specs bump **user.db v4→v5** (recursive-safety, content-variant, s7ae, config-engine, goal) and **source.db v2→v3**
  (compaction-event, ingest-fidelity, secret-redaction, zstd, blob-heartbeat). These MUST be batched: pick the coherent
  set of ready beads per tier, land them together, and rebuild the live 38 GB archive **once** (`ops reset --index` /
  the durable migration + backup manifest) — never per-isolated-addition. **Blue-green (b5l) removes the index-rebuild
  downtime, so land b5l first and every subsequent index bump is a non-event.**
- **Cross-tier VIEW constraint (recurring trap).** SQLite forbids a *persistent* `CREATE VIEW` in index.db that
  references the ATTACHed `user_tier` (user.db). Anything joining `assertions ↔ blocks/messages` (correction-edge,
  recall, annotation query-back) must be a **runtime query method** (like `query_assertions`) or a per-connection
  `TEMP VIEW` — never DDL. Guard it with a comment + a `devtools lab policy` check; a future contributor will try the VIEW.
- **Recursive-safety is one chokepoint, not per-path.** Enforce `author_kind != user ⇒ CANDIDATE + inject:false` inside
  `upsert_assertion` itself (never resurrecting a terminal-judged row), so the transform/pathology/goal/decision/recipe/
  distillery/recall writers and the `blackboard_post` agent-write path all inherit it. Enforcing it per-recipe leaves the
  `blackboard_post`-ACTIVE hole open. This is the QUOTED→OPERATOR promotion gate 37t.11's injection-security depends on.
- **Name-collision & registered-kind hygiene.** New units must avoid existing names (`raw_artifacts` = source-tier ingest
  taxonomy, not session artifacts; `threads` = lineage table, so cross-origin is `cross_origin_threads`). Every new
  `AssertionKind`/`ObjectRefKind` must be a *registered* kind (`insight:` not an invented prefix) and gets a `user_audit`
  surface entry, or the every-kind audit invariant fails.

## 5. GPT-pro prompt fan-out (66 prompts; 11 as files)

Full list + bodies in `.agent/scratch/rnd-brainstorm-2026-07-05.md` (A1-A33 design, D1-D33 deepresearch).
Files ready to paste in `/realm/inbox/tmp/`: `00-INDEX`, `A09-webui-cockpit`, `A11-corpus-compaction`, `A16-measure-registry`,
`A17-query-finding-objects`, `A18-recursive-safety`, `A20-compaction-event`, `A27-episode-merge`, `D01-competitive-landscape`,
`D02-local-llm-stack`, `D07-rl-eval-environment`. Standing depth contract (prepend to each): ground in the bundle (cite
file:line/bead), tag [evidence] vs [proposal], one defended recommendation + runner-up, three layers (today / near-term /
full direction), end with Open-questions + What's-missing, flag staleness, don't re-derive swarm2. Deepresearch lanes worth
running: competitive landscape, local-LLM+embedding stack, retrieval for long transcripts, harness+interchange-format
landscape, citation-anchor standard, cost/subscription models, RL/eval environments, agent-memory prior art, construct-
validity-as-product, verifiable-deletion, summary-faithfulness, persisted-queries-as-objects, big-graph/PE/ambient-theming.

**Flywheel correction:** the browser extension already auto-captures GPT-pro chats into the archive — there is **no
manual export/re-ingest step**. The only manual work is making each R&D batch a retrievable cohort (title stub / tag) and
distilling captured sessions into beads. Recursive-safety gates the distilled assertions.

---

## 6. The critical path (my synthesis; the prioritization agent will sharpen)

The honesty red-team's strongest argument: **~80% of realized value is "agent-memory + cost-visibility"; ~70% of the
backlog (insight museum, result-set algebra, lineage normalization, export/portfolio cluster) is orthogonal** and may be
over-engineering. Take it seriously. Ranked by (operator-value ÷ build-cost + risk):

**Tier 0 — fix the lies (days, mostly code-only, no new capability):** cost-correctness (dead exact path, missing Opus
credit rates, logical-grain double-count, tier $/credit), honesty-audit completeness, security DNS-rebinding + receiver
token, the two temporal bugs, attachment referenced-vs-stored honesty, jnj.5 mutation contract. These are pure credibility
— the brand is "every number resolves to bytes" and several numbers currently lie.

**Tier 1 — the load-bearing substrate (unblocks everything):** the `ContextSource`/37t.11 scheduler (its trust-class gate
is the recursive-safety spine); blue-green index rebuild (b5l — makes every later derived-tier bump downtime-free);
the measure-registry (turns analytics into a fill-the-registry job and absorbs the cost fixes via LOGICAL_SESSION grain).

**Tier 2 — the differentiated wedge (the launch story):** the claim-vs-evidence `.polydemo` + demo-as-CI-test + the
hero-finding README (positioning); adoption/install (`polylogue install` + `doctor` — the substrate is worthless if agents
don't use it); MCP-collapse to 9 verbs (the continuity surface an agent can actually navigate).

**Tier 3 — the convergent frontier (high-leverage, higher cost):** queries+findings-as-objects (needs set-algebra first);
the episode unit; recursive-safety subsystem; content-hash citation anchors + webui cockpit.

**Defer / question (the red-team's cut list):** the result-set algebra beyond what findings-as-objects needs; the insight
"museum" (behavioral analytics run once and screenshotted); the export/portfolio cluster; the composer TUI (swarm2) until
the daemon/thin-client is real. Build these only when a live query demands them.

**Decisions only the operator can make:** (1) product identity — OSS-personal-substrate vs product (positioning agent says
OSS-by-necessity, defer monetization); (2) whether to accept the durable user v4→v5 / source v2→v3 migrations as a batch
now; (3) whether "daemon required" (swarm2) is the target or daemonless stays first-class.

---

## 7. State / pointers

- **Sinnix:** `mcp-polylogue` archive-root-leak fix committed `f6de855` (NOT switched — dirty tree has unrelated
  uncommitted work; NOT pushed — push-after-deploy). Applying needs `nix develop --command switch` and will sweep the
  other uncommitted changes; that's the operator's call.
- **CLAUDE.md rewrite:** standalone ~4K-token version merged to master (#2546); AGENTS.md is a symlink; render-agents
  machinery removed. The `fix(readiness)` commit is #2547.
- **Master idea doc:** `.agent/scratch/rnd-brainstorm-2026-07-05.md`. **This synthesis:** `.agent/scratch/rnd-master-synthesis-2026-07-05.md`.
- **Prompt files:** `/realm/inbox/tmp/`. **Agent transcripts:** `<session>/tasks/*.output` (read one at a time).
- **Live archive gotcha:** MCP still points at `/tmp/polylogue-archive` this session (the sinnix fix isn't deployed yet) —
  for live-archive CLI use `export POLYLOGUE_ARCHIVE_ROOT=/home/sinity/.local/share/polylogue`.

## 8. Synthesis lanes — critical-path & launch verdicts (sharpen §6; folded post-hoc, hence after §7)

**Critical-path lane.** The red-team's 80/20 is *directionally right but its proposed core is too small*. The honest minimal core is NOT "SessionEnd-copy + FTS + 3 views + 1 tool" (that's `llm`-logs+grep) — it keeps the three cheap differentiators grep can't reconstruct: **`material_origin`** (makes honest cost/user-word accounting possible), the **~10 real MCP continuity tools** (save_annotation/get_resume_brief/recall — not one search tool), and **content-hash idempotency + split-tier durability** (lets the archive rebuild without losing irreplaceable user.db). What IS orthogonal and deferrable: the 16-measure "museum," result-set set-algebra, the content-variants/translation epic (cut), CIF/federation, RL-env, temporal-navigation UI. Sequence: **Weeks 1–2** recursive-safety gate (now load-bearing *safety* since the archive auto-captures its own R&D) + the confirmed bug track + activation layer; **Weeks 3–4** t46 contract-thinning + construct-validity gate + cost-per-outcome (`terminal_state × session_costs`, one materializer) + blue-green rebuilds + attachment-honesty-half; **later, gated on operator** citation-anchors + compaction-pack + findings-as-objects. Additional confirmed bugs it pinned (beyond §2): **`1xc.11` convergence probes fail-*closed* to 'converged'** (a probe error silently suspends auto-convergence — highest severity), **`a7xr.1` ~9 sqlite3 connection leaks** (`with connect()` commits-not-closes), **`4ts.6` lineage truncation at `_MAX_LINEAGE_DEPTH=64` with no completeness flag in the read envelope**.

**Whole-product/launch lane.** Identity = **OSS-by-necessity personal substrate, NOT a product** (no users/roadmap/funnel — the operator's success metric is his own daily use; a product identity distorts toward a stranger's onboarding funnel and reads as *less* serious to the fundability/hiring audience that judges code+findings). The **entire distribution stack is already built and idle** — PyPI/Homebrew/GHCR/FlakeHub/Nix all wired, but **zero git tags ever fired, version frozen 0.1.0**, and install docs say "no packaged path." So the whole remaining game is **decision, not build**: (1) fire the first release tag; (2) prove the install matrix on a cold machine + reconcile the "not available" docs; (3) de-meta the README (strip "claims stay capability-phrased until measured…" meta-sentences, replace agent-coined jargon) with every retained claim demonstrable via `demo tour` or a cited finding; (4) regenerate ONE flagship live-cited finding on v24; publish screencast (already recorded). Launch staircase: release-cutover → quiet uvx one-liner + cold-reader pass → data-story essay (spine = **forensics on real data**, e.g. the 24.1% silent-proceed finding, NOT memory-uplift — `cfk`/`jxe` uplift pilot closed *negative*, so all memory claims stay capability-phrased) → one sharp thread into the AI-memory discourse with the s7ae two-agent demo attached. First-10-users = heavy coding-agent operators + local-first/self-hosters + Claude-Code/Codex power-users. Monetization: **defer explicitly** (patronage/grant/employment-leverage; build no signup/telemetry/tiers — they contradict local-first and weaken the competence signal). Do NOT build: separate docs site as a launch gate, Datasette exhibit, interchange-schema, multi-model leaderboard, any new capability for launch.

## 9. Next actions (when you pick up)
1. Convert Tier-0 fixes to execution-grade beads and start landing them (they're code-only, high-credibility, low-risk).
2. Batch the durable migrations decision before touching user.db/source.db (see §4 "BATCH THE SCHEMA BUMPS").
3. **Provenance caveat (read §0.1):** the file:line citations in §1/§2 and the wave-4 entries in §3 are distilled from the
   pre-compaction summary, not re-read from source this session. Before acting on any specific citation, re-read that
   spec's `tasks/*.output` to confirm the file:line — the task files are the authority, this doc is the index.


=====================================================================
  PART 2 - PAYLOAD: 15 subagent outputs (unique to file 2 of 6)
=====================================================================


---

## [a49c0f4aa37924572] Wave-5 Implementation Spec: `turn-pair`, `artifact`, `correction-edge`

I have complete grounding. Here is the implementation spec for the three missing units.

---

# Wave-5 Implementation Spec: `turn-pair`, `artifact`, `correction-edge`

## Grounding facts that shape all three specs

- **`actions` is the model VIEW** (`index.py:324`): `LEFT JOIN blocks r ON r.tool_id = u.tool_id … WHERE u.block_type='tool_use'`. Pairs two block rows into one relation on read. Query units lower to SQL via a `sql_query_method` on `ArchiveTier` (`archive.py`), often wrapping the VIEW in a CTE (`_ACTION_FOLLOWUP_RELATION_SQL`, `archive.py:449`).
- **Query-unit registration is descriptor-driven** (`archive/query/metadata.py:740`): each unit is a `QueryUnitDescriptor(unit, singular, plural, payload_model, sql_query_method, aggregate_group_fields, fields, …)`. `QueryUnitName` is a closed `Literal` (`metadata.py:8`). Adding a unit = extend that Literal + append a descriptor + add `STRUCTURAL_QUERY_UNIT_REGISTRY` entry + a `payload_model` + a `query_*` method.
- **Cross-tier attach** (`connection.py:62`): `user.db` is ATTACHed into the `index.db` connection as schema alias **`user_tier`**. `query_assertions` (`archive.py:5441`) reads `FROM user_tier.assertions a LEFT JOIN sessions s ON a.target_ref = 'session:'||s.session_id` as **inline SQL, not a stored VIEW**. This is load-bearing: **SQLite forbids a persistent VIEW from referencing an ATTACHed database's tables** — so anything joining `assertions`↔`blocks` must be a runtime query method or a `TEMP VIEW`, never DDL.
- **`material_origin`** (`enums.py:176`): `human_authored, assistant_authored, operator_command, runtime_protocol, runtime_context, tool_result, generated_context_pack, generated_analysis_pack, unknown`. Indexed: `idx_messages_session_material_origin` (`index.py:153`).
- **`files` unit already exists** (`archive.py:5115`, `query_files`) — a **runtime aggregation over `actions` grouped by `(session_id, tool_path)`** yielding path + action_count + first/last block. It has **no per-operation provenance, no create-vs-edit-vs-read distinction, no artifact→artifact lineage**. This is the gap `artifact` fills — and the primary risk (see below).
- **Name collision:** `raw_artifacts` already exists in the **source tier** (`source.py:84`) meaning *ingest-taxonomy* ("is this file parseable as a session"). The new unit must **not** reuse `artifact_id`/`raw_artifacts`. Use `session_artifacts` / `artifact_lineage`.
- **ObjectRef kinds** (`refs.py:8`) already include `message`, `block`, `file`, `tool-call`. `correction` assertions set `target_ref = normalize_object_ref_text(f"{target_type}:{target_id}")` (`user_write.py:275`), e.g. `block:<block_id>` or `message:<message_id>`. That is the citation-anchor.
- **Tier/migration regimes** (CLAUDE.md): `index.db` (v24) and `embeddings.db` are **derived — no numbered migration chain**; a schema change edits canonical DDL + rebuild plan (`polylogue ops reset --index && polylogued run`). `user.db` (v4) is **durable — numbered additive migration + backup manifest**. All three units land in the **derived index tier**; none needs a user.db bump.

---

## UNIT 1 — `turn-pair`

The natural grain for per-turn latency / cost / correction-rate / answer-anchoring: one human↔assistant exchange as one row, via `material_origin` adjacency. Directly analogous to `actions` (two block rows → one relation).

### (1) Schema/DDL — tier: `index.db` (derived) · **VIEW** (like `actions`)

A stored VIEW is legal: all referenced tables (`messages`) live in `main`. Pair each prompting message (`human_authored` or `operator_command`) with the **next `assistant_authored`** message in the same session on the active path.

```sql
CREATE VIEW IF NOT EXISTS turn_pairs AS
SELECT
    p.session_id                                   AS session_id,
    p.message_id                                   AS prompt_message_id,
    p.material_origin                              AS prompt_origin,
    p.position                                     AS prompt_position,
    p.word_count                                   AS prompt_word_count,
    p.occurred_at_ms                               AS prompt_occurred_at_ms,
    a.message_id                                   AS answer_message_id,
    a.model_name                                   AS answer_model_name,
    a.model_effort                                 AS answer_model_effort,
    a.position                                     AS answer_position,
    a.occurred_at_ms                               AS answer_occurred_at_ms,
    a.output_tokens                                AS answer_output_tokens,
    a.input_tokens                                 AS answer_input_tokens,
    a.cache_read_tokens                            AS answer_cache_read_tokens,
    a.cache_write_tokens                           AS answer_cache_write_tokens,
    a.has_tool_use                                 AS answer_has_tool_use,
    a.duration_ms                                  AS answer_duration_ms,
    (a.occurred_at_ms - p.occurred_at_ms)          AS turn_latency_ms
FROM messages p
LEFT JOIN messages a
    ON  a.session_id      = p.session_id
    AND a.material_origin = 'assistant_authored'
    AND a.is_active_path  = 1
    AND a.position = (
        SELECT MIN(n.position) FROM messages n
        WHERE n.session_id = p.session_id
          AND n.material_origin = 'assistant_authored'
          AND n.is_active_path = 1
          AND n.position > p.position
    )
WHERE p.material_origin IN ('human_authored', 'operator_command')
  AND p.is_active_path = 1;
```

Supporting index (add to `index.py` DDL) — the correlated subquery needs a covering scan on `(session_id, material_origin, is_active_path, position)`; the existing `idx_messages_session_material_origin` lacks `position`/`is_active_path`:

```sql
CREATE INDEX IF NOT EXISTS idx_messages_turn_pairing
ON messages(session_id, material_origin, is_active_path, position)
WHERE is_active_path = 1;
```

- `LEFT JOIN` intentional: an **unanswered prompt** (abandoned turn) yields a row with NULL answer columns — a first-class signal (feeds `find_abandoned_sessions`).
- `turn_latency_ms` NULL when either timestamp NULL (honest-NULL discipline, mirroring `actions.exit_code`).

### (2) Derivation algorithm
Position-adjacency skip-scan: for each prompt row, `MIN(position)` among later assistant rows on the active path. Intervening `tool_result` / `runtime_protocol` / `runtime_context` rows are skipped because they don't match `material_origin='assistant_authored'`. Variant branches handled by `is_active_path=1` (deterministic single leaf). Turn cost = sum of `answer_*_tokens` (priced downstream via existing cost tables keyed by `model_name`). Answer-anchoring: `answer_message_id` is the anchor a `correction-edge` binds to (Unit 3).

### (3) Migration
Derived tier → **no numbered migration**. Edit `index.py` canonical DDL (add VIEW + index), bump `INDEX_SCHEMA_VERSION` 24→25, add a one-line rebuild-plan note. Ship: `polylogue ops reset --index && polylogued run`. Register query unit in `metadata.py`: extend `QueryUnitName` Literal with `"turn-pair"`, append a `QueryUnitDescriptor("turn-pair", "turn-pair", "turn-pairs", payload_model="TurnPairQueryRowPayload", sql_query_method="query_turn_pairs", aggregate_group_fields=("answer_model_name","prompt_origin","session.origin","session.repo"), …)`, add `STRUCTURAL_QUERY_UNIT_REGISTRY["turn-pair"]`, register `TurnPairQueryRowPayload`, implement `ArchiveTier.query_turn_pairs`. Regenerate `render openapi` + `render cli-output-schemas` + topology projection.

### (4) Test strategy
- **VIEW-correctness unit test** (new `tests/unit/storage/test_turn_pair_view.py`, non-protected): `SessionBuilder` a session with human→tool_result→tool_result→assistant→human→(no answer). Assert exactly 2 turn_pair rows; first pairs to the correct assistant skipping the two tool_result rows; second has NULL answer (abandoned).
- **Property test** (extend `tests/property`): for any built session, every `turn_pairs` row's `answer_position > prompt_position` and no assistant message is claimed by two prompts (monotone pairing).
- **Latency-sign invariant:** `turn_latency_ms >= 0` whenever both timestamps present (guards clock-skew reversals — flag as data anomaly, don't crash).
- **Query-law parity:** add turn-pair to `tests/unit/cli/test_query_exec_laws.py` / `test_verb_cardinality.py` so `find "turn-pairs where answer_model_name:…"` executes and `| group by answer_model_name | count` rolls up. Use `frozen_clock`.

### (5) Bead (draft — do NOT create)
```
bd create --type feature --title "turn-pair query unit: human↔assistant exchange as one row" \
  --body "Add `turn_pairs` VIEW (index.db, derived) pairing prompt (human_authored|operator_command)
          with next assistant_authored on active path via position adjacency. Register `turn-pair`
          query unit. The natural grain for per-turn latency/cost/correction-rate/answer-anchoring.
   AC:
   1. VIEW in index.py DDL + idx_messages_turn_pairing; INDEX_SCHEMA_VERSION bumped + rebuild-plan note.
   2. Unanswered prompt → row with NULL answer cols (abandoned-turn signal), not dropped.
   3. Intervening tool_result/runtime_* rows skipped; variant branches resolved by is_active_path=1.
   4. Query unit registered: Literal, descriptor, STRUCTURAL_QUERY_UNIT_REGISTRY, payload, query_turn_pairs.
      `find 'turn-pairs where answer_model_name:opus'` + `| group by answer_model_name | count` execute.
   5. turn_latency_ms = answer.occurred_at - prompt.occurred_at, NULL when either NULL; >=0 invariant test.
   6. render openapi + cli-output-schemas + topology projection regenerated; devtools verify green."
```

### (6) Top-3 risks
1. **Multi-answer / streamed-variant turns.** Providers emit assistant continuation variants (`variant_index>0`, retries). `is_active_path=1` + `MIN(position)` collapses to one canonical answer, but token/latency accounting silently drops sibling variants — under-counts cost on regenerated turns. Mitigation: document that turn cost = active-leaf answer only; expose variant count as a follow-up column.
2. **Operator-command noise.** `operator_command` rows (Claude Code slash-commands, `runtime_protocol` `role=user`) as prompts inflate turn count and pollute latency medians. Decision needed: include `operator_command` (current spec does) vs. `human_authored`-only. Recommend a `prompt_origin` filter so consumers choose; do **not** silently drop, per the honest-accounting doctrine.
3. **Timestamp gaps → NULL latency floods.** Web exports (`chatgpt-export`, `claude-ai-export`) frequently lack per-message `occurred_at_ms`; `turn_latency_ms` is NULL for whole origins. Latency dashboards must treat NULL as "unmeasurable", not zero — same trap as `session_latency_profiles`.

---

## UNIT 2 — `artifact`

A file a session **created / edited / read**, distinct from `files`-attachments (uploads) and from `raw_artifacts` (ingest taxonomy). Adds per-operation provenance and artifact→artifact lineage that the existing `files` aggregation lacks.

### (1) Schema/DDL — tier: `index.db` (derived) · **VIEW → table**

**Phase A (MVP, VIEW)** — per-operation artifact touches, projected from `actions` (which already exposes `tool_name`, `tool_path`, `semantic_type`, `is_error`). This is strictly richer than `files`: one row per operation, classified, not aggregated.

```sql
CREATE VIEW IF NOT EXISTS artifact_touches AS
SELECT
    a.session_id                                   AS session_id,
    a.message_id                                   AS message_id,
    a.tool_use_block_id                            AS tool_use_block_id,
    a.tool_result_block_id                         AS tool_result_block_id,
    REPLACE(a.tool_path, char(92), '/')            AS path,
    a.tool_name                                    AS tool_name,
    CASE
        WHEN lower(a.tool_name) IN ('write','create_file')                THEN 'create'
        WHEN lower(a.tool_name) IN ('edit','str_replace','apply_patch','multiedit','notebookedit') THEN 'edit'
        WHEN lower(a.tool_name) IN ('read','cat','view','open')           THEN 'read'
        ELSE COALESCE(NULLIF(a.semantic_type,''), 'touch')
    END                                            AS operation,
    a.is_error                                     AS is_error
FROM actions a
WHERE a.tool_path IS NOT NULL AND a.tool_path != '';
```

**Phase B (materialized table + lineage)** — a VIEW cannot cheaply express artifact→artifact edges (read-then-write provenance, renames/moves) nor cross-session provenance; those need a **materialized derived table** written during the `materialize` ingest stage.

```sql
CREATE TABLE IF NOT EXISTS session_artifacts (
    session_artifact_id  TEXT GENERATED ALWAYS AS (session_id || ':artifact:' || path) STORED UNIQUE,
    session_id           TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    path                 TEXT NOT NULL,
    first_operation      TEXT NOT NULL CHECK(first_operation IN ('create','edit','read','touch')),
    created_in_session   INTEGER NOT NULL DEFAULT 0 CHECK(created_in_session IN (0,1)),
    edit_count           INTEGER NOT NULL DEFAULT 0 CHECK(edit_count >= 0),
    read_count           INTEGER NOT NULL DEFAULT 0 CHECK(read_count >= 0),
    error_count          INTEGER NOT NULL DEFAULT 0 CHECK(error_count >= 0),
    first_tool_use_block_id TEXT,
    last_tool_use_block_id  TEXT,
    first_seen_ms        INTEGER,
    last_seen_ms         INTEGER,
    PRIMARY KEY(session_id, path)
) STRICT;

CREATE TABLE IF NOT EXISTS artifact_lineage (
    src_session_artifact_id TEXT NOT NULL REFERENCES session_artifacts(session_artifact_id) ON DELETE CASCADE,
    dst_session_artifact_id TEXT NOT NULL REFERENCES session_artifacts(session_artifact_id) ON DELETE CASCADE,
    edge_kind               TEXT NOT NULL CHECK(edge_kind IN ('read-into-write','rename','copy','derives')),
    evidence_block_id       TEXT,
    confidence              REAL NOT NULL DEFAULT 1.0 CHECK(confidence BETWEEN 0 AND 1),
    PRIMARY KEY(src_session_artifact_id, dst_session_artifact_id, edge_kind)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_artifacts_path ON session_artifacts(path);
CREATE INDEX IF NOT EXISTS idx_artifact_lineage_dst   ON artifact_lineage(dst_session_artifact_id);
```

The public `artifact` query unit reads `session_artifacts` (Phase B) or `artifact_touches` (Phase A). Keep the unit token `artifact`/`artifacts`.

### (2) Derivation algorithm
During `materialize` (in `pipeline/services/ingest_batch/`, alongside where `session_profiles` are built), scan `actions` for the session ordered by block position:
- classify each op (create/edit/read) by `tool_name` (table above);
- `created_in_session=1` iff the **first** op on that path is `create` (or an `edit` with no prior `read`/`create` where provider signals new-file);
- accumulate `edit_count`/`read_count`/`error_count`, `first/last_seen_ms`, `first/last_tool_use_block_id`.
- **Lineage** (`read-into-write`): within a session, if path B is `create`/`edit` at position q and path A was `read` at position p<q within the same assistant turn (share `answer_message_id` from Unit 1's turn_pairs), emit `read-into-write` edge A→B (`confidence` scaled by turn distance). `rename`/`copy` inferred from bash `mv`/`cp` `tool_command` argument parsing (low-confidence, evidence_block_id set). Cross-session `derives`: same absolute path edited in a later session → optional Phase C, out of MVP scope.

### (3) Migration
Derived tier → edit `index.py` DDL (VIEW for Phase A; two tables for Phase B), bump `INDEX_SCHEMA_VERSION`, add materialize-stage writer + rebuild-plan note, `ops reset --index && polylogued run`. Register `artifact` query unit in `metadata.py` (add to `PROJECTION_QUERY_UNITS` if it should support `exists artifact(...)` selectors, mirroring `file`). Regenerate generated surfaces + topology projection (new module under `polylogue/` → topology projection **must** be regenerated or `render all --check` fails).

### (4) Test strategy
- **Provenance-classification unit test:** build a session with Read(f), Edit(f), Write(g); assert `session_artifacts` has f(read_count=1,edit_count=1,created_in_session=0) and g(created_in_session=1,first_operation=create).
- **Lineage test:** Read(a.py) then Write(b.py) in one assistant turn → one `read-into-write` edge a→b; separate turns → no edge (or lower confidence).
- **Distinctness invariants (critical, guards the overlap risk):** (a) `session_artifacts` rows never include `attachments` upload paths; (b) a session with only uploads and no tool ops yields zero artifacts; (c) `raw_artifacts` (source tier) is untouched.
- **Idempotency:** re-materialize same session → identical `session_artifacts` (content-hash-stable), no duplicate lineage edges.
- **Rebuild-parity:** `ops reset --index` + reingest reproduces byte-identical artifact tables (derived-tier contract).

### (5) Bead (draft — do NOT create)
```
bd create --type feature --title "artifact query unit: session-created/edited/read files with provenance + lineage" \
  --body "Model files a session touched (create/edit/read) with per-op provenance and artifact→artifact
          lineage. Distinct from `files` (aggregation, no provenance), `attachments` (uploads), and source-tier
          `raw_artifacts` (ingest taxonomy — DO NOT reuse that name).
   AC:
   1. Phase A: `artifact_touches` VIEW over actions (per-op, classified create/edit/read/touch).
   2. Phase B: `session_artifacts` + `artifact_lineage` materialized tables (index.db, derived), written in
      materialize stage next to session_profiles; STRICT; created_in_session/edit_count/read_count/error_count.
   3. read-into-write lineage edge when read(A) and write/edit(B) share an assistant turn (turn_pairs).
   4. Distinctness tests: excludes attachment uploads; empty when session has only uploads; raw_artifacts untouched.
   5. `artifact`/`artifacts` query unit registered (+ PROJECTION_QUERY_UNITS if exists-selector wanted);
      `find 'artifacts where operation:edit AND path:archive'` + group-by-path count execute.
   6. Idempotent re-materialize; ops reset --index rebuild-parity; topology projection + generated surfaces regenerated."
```

### (6) Top-3 risks
1. **Overlap with existing `files` unit → redundant/confusing surface.** `files` (`archive.py:5115`) already answers "which paths did this session touch." If `artifact` isn't clearly the provenance/lineage superset, it duplicates. Mitigation: position `artifact` as the per-operation + lineage grain and consider deprecating `files` into an `artifact | group by path` rollup in a follow-up — but **don't** silently break `files` consumers (`filter_builder.py:131` `path:` predicate, `tool_usage.py`).
2. **Tool-name classification is provider-specific and open-ended.** `Write/Edit/Read` is Claude Code; Codex uses `apply_patch`; ChatGPT canvas, Gemini, bash `>`/`sed`/`tee` redirections don't surface as typed tools. The `operation` CASE will mislabel or bucket to `touch`, corrupting create-vs-edit rates. Mitigation: drive classification from a small provider→operation mapping table reviewed against the live corpus (`mcp__polylogue__tool_usage`), and keep `touch` as an honest unknown rather than guessing.
3. **Name/identity collision & lineage cost.** `session_artifact_id` generated from `path` breaks if the same path is touched via absolute + relative forms; NFC/backslash normalization must match `files` (`REPLACE(tool_path, char(92), '/')`). Lineage inference (`read-into-write`, `mv` parsing) is heuristic — over-eager edges create false provenance. Ship lineage confidence-scored and off the critical read path; validate edge precision on real data before any UI leans on it.

---

## UNIT 3 — `correction-edge`

A `CORRECTION` assertion bound to the exact block/turn it corrects, resolving `target_ref` (citation-anchor) → block/message → tool/model. Enables error-rate-per-tool and error-rate-per-model.

### (1) Schema/DDL — tier: cross-tier read (`user_tier.assertions` ⋈ `index.db`) · **runtime query method, NOT a stored VIEW**

**Hard constraint:** a persistent `CREATE VIEW` in `index.db` cannot reference `user_tier.assertions` (ATTACHed DB) — SQLite rejects it. Therefore `correction-edge` follows the **exact pattern of `query_assertions`** (`archive.py:5393`): an inline SQL join executed after `_attach_user_tier_if_present()`. (If a VIEW is desired for ergonomics, it must be a per-connection `TEMP VIEW` created post-attach — spec the query-method as primary.)

Inline SQL (implemented as `ArchiveTier.query_correction_edges`):

```sql
-- after self._attach_user_tier_if_present()
WITH corrections AS (
    SELECT
        c.assertion_id,
        c.target_ref,
        c.body_text,
        c.value_json,
        c.author_kind,
        c.status,
        c.created_at_ms,
        -- citation-anchor parse: 'block:<id>' | 'message:<id>' | 'session:<id>'
        substr(c.target_ref, 1, instr(c.target_ref, ':') - 1)  AS anchor_kind,
        substr(c.target_ref, instr(c.target_ref, ':') + 1)      AS anchor_id
    FROM user_tier.assertions c
    WHERE c.kind = 'correction' AND c.status = 'active'
)
SELECT
    x.assertion_id,
    x.target_ref,
    x.anchor_kind,
    x.body_text,
    x.author_kind,
    x.created_at_ms,
    b.block_id                                     AS corrected_block_id,
    b.tool_name                                    AS corrected_tool_name,
    b.block_type                                   AS corrected_block_type,
    b.tool_result_is_error                         AS corrected_is_error,
    m.message_id                                   AS corrected_message_id,
    m.material_origin                              AS corrected_material_origin,
    m.model_name                                   AS corrected_model_name,
    s.session_id                                   AS session_id,
    s.origin                                       AS origin
FROM corrections x
LEFT JOIN blocks   b ON x.anchor_kind = 'block'   AND b.block_id   = x.anchor_id
LEFT JOIN messages m ON m.message_id = COALESCE(
                            b.message_id,
                            CASE WHEN x.anchor_kind = 'message' THEN x.anchor_id END)
LEFT JOIN sessions s ON s.session_id = COALESCE(
                            m.session_id,
                            CASE WHEN x.anchor_kind = 'session' THEN x.anchor_id END)
WHERE {predicate_clause}
ORDER BY x.created_at_ms DESC, x.assertion_id
LIMIT ? OFFSET ?;
```

`corrected_model_name` comes from the message; for a block anchored to a `tool_result`, `corrected_tool_name`/`corrected_is_error` give the tool axis → the row is directly groupable for **error-rate-per-tool** and **error-rate-per-model**.

### (2) Derivation algorithm
1. Read `assertions WHERE kind='correction' AND status='active'` from `user_tier`.
2. Parse the citation-anchor: split `target_ref` on first `:` → `(anchor_kind, anchor_id)` using the same `ObjectRef` grammar (`refs.py`, `normalize_object_ref_text`).
3. Resolve by anchor kind: `block:` → `blocks` (then up to its message/session); `message:` → `messages`; `session:` → `sessions` (coarse, no tool/model). `turn:`/`tool-call:` anchors (future) resolve via `turn_pairs.answer_message_id` (Unit 1) — spec the join, gate behind Unit 1 landing.
4. Emit one edge per correction; unresolved anchors (target block GC'd or not yet ingested) yield NULL join columns — surfaced as `resolution:unresolved`, never dropped (mirrors `session_links` unresolved discipline).
5. **Error-rate rollup** is then a query-unit aggregate: `correction-edges | group by corrected_tool_name | count` over a denominator of `actions | group by tool_name | count`.

### (3) Migration
**No schema change to any tier.** `assertions` (user.db v4) already stores corrections with `target_ref`; `blocks`/`messages` exist. This is a **pure read-model / query-unit addition**:
- Register `correction-edge` in `metadata.py` (`QueryUnitName` Literal, descriptor with `sql_query_method="query_correction_edges"`, `lowerer_kind="sql"`, `aggregate_group_fields=("corrected_tool_name","corrected_model_name","anchor_kind","session.origin")`, `STRUCTURAL_QUERY_UNIT_REGISTRY` entry).
- Add `CorrectionEdgeQueryRowPayload` + `ArchiveTier.query_correction_edges`.
- Regenerate `render openapi` + `render cli-output-schemas` + topology projection.
No numbered migration, no `ops reset` required for the data (it's read-through) — but a topology-projection regen is mandatory (new module/method).

### (4) Test strategy
- **Anchor-resolution unit test:** insert a `correction` assertion with `target_ref='block:<known_block_id>'`; assert the edge resolves `corrected_tool_name`/`corrected_model_name` correctly; a `message:` anchor resolves without a block; a `session:` anchor resolves coarse (NULL tool).
- **Unresolved-anchor test:** `target_ref='block:ghost'` → one row, all corrected_* NULL, `resolution:unresolved` — not dropped.
- **Error-rate-per-tool law:** seed N tool_result blocks for `bash`, correct 2 of them, assert `correction-edges where corrected_tool_name:bash | count` = 2 and the rate against `actions` denominator is 2/N.
- **Cross-tier attach guard:** run against an archive with `user.db` absent → returns `[]` (mirror `query_assertions` `if not self.user_db_path.exists(): return []`, `archive.py:5405`).
- **Kind isolation:** non-`correction` assertions (mark/tag/annotation) must not leak into the relation. Use `frozen_clock`; add to `test_query_exec_laws.py`.

### (5) Bead (draft — do NOT create)
```
bd create --type feature --title "correction-edge query unit: CORRECTION assertion bound to corrected block/turn" \
  --body "Join user_tier.assertions(kind=correction) to blocks/messages via target_ref citation-anchor,
          resolving corrected tool_name/model_name → enables error-rate-per-tool and error-rate-per-model.
          MUST be a runtime query method (like query_assertions), NOT a stored VIEW: SQLite forbids a
          persistent VIEW referencing the ATTACHed user_tier DB.
   AC:
   1. query_correction_edges inline SQL after _attach_user_tier_if_present(); no DDL VIEW across user_tier.
   2. Anchor parse (block:/message:/session:) via ObjectRef grammar; block→message→session resolution.
   3. Unresolved anchor (GC'd/not-ingested target) → row with NULL corrected_*, resolution:unresolved, not dropped.
   4. user.db absent → returns []. Only kind='correction' status='active' rows included (kind isolation test).
   5. `correction-edge` unit registered (Literal, descriptor, STRUCTURAL registry, payload); group-by
      corrected_tool_name/corrected_model_name | count executes; error-rate-per-tool law test passes.
   6. No tier schema change (pure read-model); topology projection + openapi + cli-output-schemas regenerated.
   Depends-on: turn: anchor resolution gated behind turn-pair unit landing."
```

### (6) Top-3 risks
1. **Anchor-resolution fragility across the derived-tier rebuild.** `block_id` is a generated column stable under deterministic re-ingest, but any `ops reset --index && polylogued run` that changes block positions (parser change, lineage recomposition) **orphans** correction anchors — the durable user-tier `target_ref` points at a block_id the rebuilt index no longer contains. This is the same class of hazard the CLAUDE.md calls out for `branch_point_message_id`. Mitigation: `resolution:unresolved` must be first-class; consider a repair pass mapping stale anchors by `(session_id, position)` fallback.
2. **Anchor granularity vs. the tool/model axis.** Most `record_correction` calls today anchor at `session:` or `message:` (per `user_write.py` envelopes), not `block:`. A `session:` anchor cannot attribute the error to a tool or model — the headline error-rate-per-tool metric degrades to "sessions with a correction." Mitigation: don't over-claim; report resolution granularity (block/message/session) alongside every rate, and push `record_correction`/MCP to prefer `block:`/`turn:` anchors.
3. **Cross-tier VIEW temptation & attach ordering.** A future contributor will try to `CREATE VIEW correction_edges` for symmetry with `actions`/`turn_pairs` and hit the ATTACHed-DB rejection at rebuild time — or a `TEMP VIEW` that silently returns empty when `user.db` isn't attached. Mitigation: encode the "runtime method, not DDL VIEW" decision in a code comment and a `devtools lab policy` check; guard on `user_db_path.exists()` exactly as `query_assertions` does.

---

## Cross-cutting notes for the implementer
- **All three units extend the same closed machinery:** `QueryUnitName` Literal + `QUERY_UNIT_DESCRIPTORS` + `STRUCTURAL_QUERY_UNIT_REGISTRY` + a `*QueryRowPayload` + an `ArchiveTier.query_*` method (`archive/query/metadata.py`, `storage/sqlite/archive_tiers/archive.py`). Miss any one and `render cli-output-schemas` / discovery tests fail.
- **Tiering summary:** turn-pair = index.db VIEW; artifact = index.db VIEW→materialized table (materialize stage); correction-edge = cross-tier runtime read (no schema change). Only turn-pair and artifact bump `INDEX_SCHEMA_VERSION` and require `ops reset --index` rebuild; none touches the durable user.db.
- **Mandatory regen** for every unit (new module/method under `polylogue/`): `devtools render topology-projection && devtools render topology-status` then `devtools render all --check` (grep for `out of sync`, don't trust the tail line), plus `render openapi` + `render cli-output-schemas`.
- **Verification:** `devtools test <new test files>` (testmon-affected), not blanket directory runs; `mypy --strict` is the net for the Literal/descriptor wiring.

**Files that ground this spec:** `polylogue/core/enums.py:176` (MaterialOrigin), `polylogue/core/refs.py:8` (ObjectRefKind), `polylogue/storage/sqlite/archive_tiers/index.py:92,182,324,521` (messages/blocks/actions-VIEW/attachments), `polylogue/storage/sqlite/archive_tiers/user.py:12` (assertions DDL), `polylogue/storage/sqlite/archive_tiers/source.py:84` (raw_artifacts collision), `polylogue/storage/sqlite/archive_tiers/archive.py:449,4946,5115,5393` (followup CTE, query_actions/files/assertions), `polylogue/storage/sqlite/archive_tiers/user_write.py:174,267` (correction target_ref), `polylogue/storage/sqlite/connection.py:62` (user_tier ATTACH), `polylogue/archive/query/metadata.py:8,740` (unit registry).

---

## [a180bbf1064a99b7d] Wave-5 Implementation Spec — Agent Coordination Substrate (epic `polylogue-s7ae`)

I have complete grounding. Writing the spec.

---

# Wave-5 Implementation Spec — Agent Coordination Substrate (epic `polylogue-s7ae`)

## 0. Grounding: what exists vs. net-new

**Already shipped** (verified in source; beads `s7ae.1`, `s7ae.4`, `bby.9` = ✓):
- `polylogue/coordination/{envelope,payloads,rendering}.py` — read-only `build_coordination_envelope(view, cwd, limit, runner)`. The **degradation ladder is already implemented** in `_work_item_payload`: Beads (`.beads/` → `bd list --json`, conf 0.95) → git branch (conf 0.35) → none (conf 0.15), each carrying `CoordinationProvenancePayload{source,command,path,confidence,freshness}`. Session-tree/topology/proof/activity/context-flow composition over `index.db` is done (`_archive_evidence_payloads`).
- CLI `polylogue agents {status,self,work-item,current,conflicts,overlap,handoff}` (`cli/commands/agents.py`), JSON-first.
- Daemon `/api/agents/coordination` route + `web_shell_coordination.py` projection + SSE at `/api/events` (`daemon/http.py`, `route_contracts.py`).
- Blackboard v1: `archive/blackboard.py` body-codec over `user.db` `assertions` (`AssertionKind.NOTE`), closed `BLACKBOARD_KINDS`, `scope_repo/issue/path/related_sessions` encoded in body text; MCP `blackboard_post`/`blackboard_list`.

**Net-new for this spec** (open beads `s7ae.3`, `s7ae.5`, dep `37t.11`):
1. **Blackboard v2 → coordination message bus**: scoped addressing, per-recipient unread/ack, TTL, `query:<hash>` live-query refs. Currently blackboard notes have no addressing scope beyond `scope_repo`, no delivery/read state, no query refs.
2. **Scheduler-mediated advisories**: today `_advisories()` returns ad-hoc strings assembled *inside* the envelope. Must move to the `37t.11` `ContextSource` protocol (bounded, trust-classed, ledgered).
3. **Two-agent live proof** (`s7ae.5`).

The DDL/algorithms below target **only the net-new**; I flag where I reuse shipped machinery.

---

## 1. Schema / DDL — tier + regime

**Tier: `user.db` (durable, irreplaceable). Regime: additive numbered migration + backup manifest, one `PRAGMA user_version` step (4→5).** Messages are user-authored epistemic claims about coordination — they belong with `assertions`, not in rebuildable `index.db`.

### 1a. No new table — reuse the unified `assertions` table

`assertions.kind` is `TEXT` with **no CHECK** (by design, so vocabulary grows without a user-tier bump — confirmed in `user.py` DDL + `AssertionKind` docstring). Two new kinds, added to `AssertionKind` (`core/enums.py`) only:

```
COORDINATION_MESSAGE = "coordination_message"   # a posted, scoped message
COORDINATION_ACK      = "coordination_ack"       # one recipient acknowledged one message
```

Column mapping for a message row (all existing columns):

| assertions column | coordination-message meaning |
|---|---|
| `assertion_id` | `_deterministic_id("coordination-message", scope_ref, author_ref, body, nonce)` |
| `kind` | `coordination_message` |
| `target_ref` | **addressing scope** as an `ObjectRef`: `repo:<root>` \| `work_item:<beads/s7ae>` \| `session:<id>` \| `agent:<codex>` \| `branch:<name>` \| `broadcast:<repo>` |
| `scope_ref` | author's session ref (`session:<id>`) |
| `body_text` | message content (reuses `build_blackboard_body` codec, `kind` ∈ extended `BLACKBOARD_KINDS`) |
| `value_json` | `{scope_kind, expires_at_ms, priority, query_refs:[...], surface_paths:[...]}` |
| `author_ref` / `author_kind` | `agent:<kind>:<pid-or-session>` / `agent` |
| `evidence_refs_json` | `query:<hash>` refs + related `session:`/`run:` refs |
| `context_policy_json` | `{"inject":true,"trust":"OPERATOR"}` for agent-authored directives; default `{"inject":false}` |
| `status` | `active` \| `expired` \| `superseded` (`AssertionStatus`) |

An **ack** row: `kind=coordination_ack`, `target_ref=assertion:<message_id>`, `author_ref=agent:<recipient>`. `unread(agent)` = in-scope active messages with no ack row by that agent.

### 1b. The one additive DDL change (migration 005) — expiry generated column + index

TTL delivery needs indexable expiry. `value_json.expires_at_ms` is not indexable as-is; STRICT tables permit a virtual generated column:

```sql
-- migrations/user/005_coordination_message_expiry.sql
ALTER TABLE assertions
  ADD COLUMN expires_at_ms INTEGER
  GENERATED ALWAYS AS (json_extract(value_json, '$.expires_at_ms')) VIRTUAL;

CREATE INDEX IF NOT EXISTS idx_assertions_coord_scope_live
  ON assertions(kind, target_ref, status, expires_at_ms);

PRAGMA user_version = 5;
```

Virtual generated columns are additive/non-rewriting (no table copy), satisfy STRICT, and index cleanly. `NULL expires_at_ms` = non-expiring. This is the **only** schema-touching change; the two new `AssertionKind`s and the `query` ObjectRef kind are Python-vocabulary edits.

### 1c. `query:<hash>` ref — the notepad→task-bus mechanism

Add `"query"` to `ObjectRefKind` + `_OBJECT_REF_KINDS` (`core/refs.py`). A `query:<sha256-12>` ref is backed by a **`saved_view` assertion** (already exists: `upsert_saved_view`, sha256 hashing in `user_write.py`). Post-flow: normalize the DSL expression → `sha256` → upsert `saved_view` under that hash → embed `query:<hash>` in the message's `evidence_refs_json`. `resolve_ref("query:<hash>")` → load saved_view → execute the DSL (`archive/query/expression.py`) → **live result set**. This is what makes a note a live task-bus entry ("sessions touching `file:archive.py` since 1h") rather than a frozen list.

---

## 2. Coordination algorithms (pseudocode)

### 2a. `post_coordination_message`

```
post_coordination_message(author, scope_kind, scope_id, kind, title, content,
                          query_exprs=[], surface_paths=[], ttl_ms=None, inject=False):
    assert scope_kind in {repo, work_item, session, agent, branch, broadcast}
    target_ref = ObjectRef(scope_kind_to_refkind[scope_kind], scope_id)   # validated, closed vocab
    query_refs = []
    for expr in query_exprs:
        normalized = normalize_dsl(expr)                # archive/query/expression.py
        h = sha256_12(normalized)
        upsert_saved_view(name=f"coord/{h}", expression=normalized)   # shipped
        query_refs.append(f"query:{h}")
    body = build_blackboard_body(kind, title, content, scope_repo=repo_root)  # shipped codec
    value = {scope_kind, expires_at_ms: now()+ttl_ms if ttl_ms else None,
             priority, query_refs, surface_paths}
    policy = {"inject": inject, "trust": "OPERATOR" if inject else "SYSTEM"}
    upsert_assertion(kind=COORDINATION_MESSAGE, target_ref, scope_ref=author.session_ref,
                     body_text=body, value_json=value, author_ref=author.agent_ref,
                     evidence_refs=query_refs + author.session_refs, context_policy=policy,
                     status=ACTIVE)
    return message_ref
```

### 2b. `list_messages_for_agent` — scoped delivery (Beads-free)

```
scopes_for(agent, repo, work_item, session_tree):
    yield ("repo", repo.root)                                  # git — always available
    yield ("branch", repo.branch)
    if work_item.ref:      yield ("work_item", work_item.ref)  # beads|git|inferred
    yield ("agent", agent.kind); yield ("agent", agent.agent_ref)
    for node in session_tree.nodes: yield ("session", node.session_id)
    yield ("broadcast", repo.root)

list_messages_for_agent(agent, envelope):
    live = SELECT * FROM assertions
           WHERE kind=COORDINATION_MESSAGE AND status=ACTIVE
             AND (expires_at_ms IS NULL OR expires_at_ms > now())
             AND target_ref IN (refs of scopes_for(agent,...))     # uses idx_..._coord_scope_live
    acked = SELECT target_ref FROM assertions
            WHERE kind=COORDINATION_ACK AND author_ref=agent.agent_ref
    unread = [m for m in live if m.id not in acked]
    addressed = [m for m in unread if m.scope_kind in {agent, session}]  # direct = highest signal
    return {unread, addressed, recent: live[:limit]}
```

**Delivery is pure query over durable rows — no push, no chatroom.** "Arrival" = the recipient's next envelope build resolves overlapping scopes. This is why it degrades: repo/branch/session scopes need only git+archive; `work_item` scope enriches when Beads is present but is never required.

### 2c. Coordination as a `ContextSource` (37t.11) — scheduler-mediated advisory

The coordination source **proposes**; it never assembles context itself. Trust class is a type-level property of the source.

```
class CoordinationContextSource(ContextSource):
    moments = {SESSION_START, MID_SESSION_ADVISORY, ON_DEMAND}

    def propose(ctx) -> list[Candidate]:
        env = build_coordination_envelope(view="status", cwd=ctx.cwd)   # shipped
        msgs = list_messages_for_agent(ctx.agent, env)
        out = []
        # 1. Direct/addressed messages — agent-authored → may instruct
        for m in msgs.addressed:
            out.append(Candidate(content_or_ref=m.ref, body=fenced(m),
                                 trust=OPERATOR if m.inject else QUOTED,
                                 token_cost=est(m), score=0.9, expiry=m.expires_at_ms,
                                 source_ref=m.ref, degrade=[full, ref_only, drop]))
        # 2. Structural facts — machine-composed, never directives
        for ov in env.overlaps:                 # same-repo-agent, resource-episode (non-blocking)
            out.append(Candidate(ref=ov.refs, body=summary(ov), trust=SYSTEM,
                                 score=0.5 if ov.severity=="info" else 0.7, ...))
        for caveat in stale_root_daemon_hook_caveats(env):     # d1y liveness
            out.append(Candidate(trust=SYSTEM, score=0.6, ...))
        return out
    # SCHEDULER (37t.11) owns: moment budget, class proportions, cross-source dedup by
    # content-hash/source_ref, cooldown, and the LEDGER row per admission decision.
    # MID_SESSION budget = 1 item → only a direct message or a high-value material change surfaces.
```

Hooks (`d1y`) call **one** scheduler entrypoint at SessionStart; they capture presence facts silently (write an `agent`-scoped assertion / touch liveness) and never emit their own advisory text. Same-file editing surfaces as a `SYSTEM` overlap candidate (`blocking=False`), competing for budget like anything else — awareness, not a gate.

### 2d. Envelope extension

Add `messages: CoordinationMessagesPayload{unread, addressed, recent}` and replace the string-heuristic `advisories` with `advisory_refs` (the scheduler ledger rows for this session) on `AgentCoordinationPayload`. `project_coordination_envelope` bounds `messages` per view (drop for `handoff`, keep for `status`/`self`/`conflicts`). Wire into CLI (`agents messages`, `agents post`), MCP (`coordination_post_message`, `coordination_inbox` — update `EXPECTED_TOOL_NAMES` + tool contract), daemon `/api/agents/coordination` (already renders envelope → free).

---

## 3. Migration

- **File**: `polylogue/storage/sqlite/migrations/user/005_coordination_message_expiry.sql` (§1b).
- **Regime gate**: durable tier → `devtools lab policy schema-versioning` requires numbered additive SQL + `USER_SCHEMA_VERSION` bump (4→5 in `user.py`) + a **verified backup manifest** before apply. No upgrade helper, no destructive step (pure `ADD COLUMN` virtual + `CREATE INDEX`).
- **Backfill**: none — new kinds, no existing rows. Legacy `AssertionKind.NOTE` blackboard rows remain valid (v1 notes read unchanged; they simply carry no addressing scope / ack semantics).
- **Non-DDL vocab edits** (no user-tier bump): `AssertionKind += {COORDINATION_MESSAGE, COORDINATION_ACK}`; `ObjectRefKind += "query"`; `BLACKBOARD_KINDS += ("advisory",)` if needed.
- **Regenerate**: new `AssertionKind`s are embedded in `render openapi` + `render cli-output-schemas` (known gotcha) → run `devtools render all --check` and grep for `out of sync`. New MCP tools → `EXPECTED_TOOL_NAMES` + tool contract. New module (if `coordination/messages.py`) → `devtools render topology-projection && topology-status`.

---

## 4. Test strategy + two-agent proof

### 4a. Unit / property (testmon-affected, per `s7ae.3` AC)

- **Direct delivery**: message scoped `agent:codex` appears in codex's `addressed`, absent from an unrelated agent's inbox.
- **Repo / work-item scoped delivery**: posted to `repo:<root>` visible to both agents in that root; `work_item:<beads-id>` visible only with matching current work item.
- **TTL/expiry boundedness**: `frozen_clock` (mandatory — `verify-test-clock-hygiene` lint) advanced past `expires_at_ms` → message drops from `live`; index predicate exercised.
- **Ack semantics**: after `coordination_ack`, message leaves `unread`, stays in `recent`.
- **Same-surface overlap is non-blocking**: assert `overlap.blocking == False` and that it appears as a `SYSTEM` candidate, never a hard stop.
- **No noisy injection**: envelope with no messages + no material overlap → scheduler proposes zero mid-session candidates (ledger empty).
- **Degradation**: build envelope in a git-only repo (no `.beads/`) → messages still deliver via repo/branch/session scopes; `work_item.source == "git"`.
- **Trust invariant** (37t.11 constraint): property test — assembled preamble never contains unfenced `QUOTED` content; red-team message body containing an injection string never reaches an assembled preamble unfenced.
- **`query:<hash>` round-trip**: post with a DSL expr → `resolve_ref` returns the live result set matching a direct DSL execution.

### 4b. Two-agent same-repo proof (`s7ae.5`, the acceptance bar)

Mirror `devtools/degraded_archive_proof.py` (`run_ln_proof(out_dir) -> dataclass.to_payload()`, JSON+MD artifacts, wired via `devtools workspace <proof>` in `command_catalog.py`, test in `tests/unit/devtools/`). **Runs on synthetic fixtures only — no private corpus.**

```
run_two_agent_coordination_proof(out_dir):
    root = git_init(tmp);  seed_demo_archive(archive_root)     # private-data-free
    wt_A = worktree(root, "feature/agent-a");  wt_B = worktree(root, "feature/agent-b")
    envA0 = build_coordination_envelope(cwd=wt_A);  envB0 = build_coordination_envelope(cwd=wt_B)

    # (a) mutual awareness: inject two fake agent processes (runner stub → ps rows) so
    #     each envelope shows the other as same-repo peer + a resource episode (non-blocking)
    envA1 = build(..., runner=stub_ps([agentB_proc, pytest_proc]))
    assert any(o.kind=="same-repo-agent" for o in envA1.overlaps)
    assert all(not o.blocking for o in envA1.overlaps)

    # (b) scoped message round-trip through the real store (no fabricated delivery)
    msg = post_coordination_message(author=A, scope_kind="agent", scope_id="agent-b",
             kind="handoff", title="taking archive.py", query_exprs=['file:archive.py since:1h'])
    envB2 = build_coordination_envelope(cwd=wt_B)
    assert msg.ref in [m.ref for m in envB2.messages.addressed]     # observed as delivered
    ack(B, msg);  envB3 = build(...);  assert msg.ref not in envB3.messages.unread

    # (c) context injection via the 37t.11 ledger
    ledger = run_scheduler(SESSION_START, agent=B)                 # coordination source proposes
    assert any(row.source=="coordination" and row.included for row in ledger.rows)

    # (d) handoff packet produced by A, referenced by both
    write(wt_A/".agent/conductor-devloop/HANDOFF-LATEST.md", ...)
    assert handoff_ref in build(cwd=wt_A).handoff and in build(cwd=wt_B).handoff

    dump(out_dir, {envA0, envB0, envA1, envB2, envB3, ledger, artifact.md})
    return DeterministicProofResult(ok=all_asserts, ...)
```

Capture **before/after `polylogue agents status --json`** from both worktrees as the evidence artifact; one documented command (`devtools workspace two-agent-coordination-proof`). Explicitly mark the epic's headline live-proof line satisfied by this committed artifact.

---

## 5. Bead breakdown (children under `s7ae.3` / `s7ae.5`)

| # | Bead (proposed) | Size | Acceptance |
|---|---|---|---|
| 1 | **Coordination message store on assertions** (`coordination_message`/`coordination_ack` kinds; post/list/ack over `user_write.py`) | M | Post→list→ack round-trips through `user.db`; scoped `target_ref`; unread = in-scope active un-acked; degrades to git-only scopes with no `.beads/`. Unit tests 4a. |
| 2 | **Migration 005 + expiry index** (virtual generated `expires_at_ms`, `idx_assertions_coord_scope_live`, `USER_SCHEMA_VERSION`→5, backup manifest) | S | `schema-versioning` policy passes; TTL query uses the index; expired messages drop under `frozen_clock`. |
| 3 | **`query:<hash>` ref kind + saved_view backing** (`ObjectRefKind += "query"`, `resolve_ref` executes DSL live) | S | `query:<hash>` in a message resolves to a live result set == direct DSL run; ref parses/formats round-trip. |
| 4 | **CoordinationContextSource (register on 37t.11)** — propose direct msgs (OPERATOR/QUOTED) + structural overlaps/caveats (SYSTEM); no self-assembly | M | Source proposes; scheduler admits within moment budget; ledger row per decision; unfenced-QUOTED property test + red-team fixture pass; mid-session budget=1. **Blocked on `37t.11` slice 1.** |
| 5 | **Envelope + surface wiring** (`messages` payload, `agents messages`/`agents post` CLI, `coordination_post_message`/`coordination_inbox` MCP, daemon route reuse) | M | Envelope carries `{unread,addressed,recent}`; `EXPECTED_TOOL_NAMES`+contract updated; `render all --check` clean (grep `out of sync`); view-bounding correct. |
| 6 | **Advisory boundedness + hook silence** (`d1y` liveness writes facts only; visible advisories only via scheduler) | S | Test: no material signal → zero mid-session injection; direct message → exactly one advisory item; hooks emit no context text. |
| 7 | **Two-agent same-repo proof** (`s7ae.5`) — `devtools/two_agent_coordination_proof.py` + run script + captured JSON | L | One documented command; both envelopes show peer+overlap+resource; one scoped message delivered+addressed; ledger context-injection row; handoff packet referenced by both; epic live-proof line marked satisfied. **Deps: 1–6 + `37t.11`.** |
| 8 | **MCP/hook pre-deploy batch** (`s7ae.2`) — complete all MCP-related code/config/tests before any switch is requested | M | Per epic AC: all program MCP surfaces recorded; if deploy is the only remaining step, note it in `s7ae` and move on. |

Ordering: 2→1→3 (storage), 4 gated on `37t.11` slice 1, 5→6 (surfaces), 7 last, 8 as the deploy gate.

---

## 6. Top-3 risks

1. **Hard dependency on unbuilt `37t.11`.** Beads 4 and 7 cannot land honestly without the ContextSource protocol + ledger, and `37t.11` carries a *blocking* injection-trust security constraint (no unfenced QUOTED, no verbatim tool/web text at any class). Building the coordination advisory path before the scheduler exists = shipping the exact injection hole the constraint warns against. **Mitigation**: sequence `37t.11` slice 1 first; until then, coordination advisories stay in the read-only envelope as SYSTEM-class structural facts only (no OPERATOR injection), and beads 1–3 (pure storage/refs) proceed independently.

2. **Per-recipient read-state in a single-row assertions model.** Unread is computed via separate `coordination_ack` rows; a broadcast message to N agents yields N ack rows and O(scopes) inbox queries per envelope build. At agent scale this stays small, but a naive `target_ref IN (...)` with many session-tree scopes could bloat. **Mitigation**: the `idx_assertions_coord_scope_live` covering index + bounding session-tree scopes to the resolved target tree (already limited in `_session_tree_payload`); cap inbox scope count; ack rows are cheap and durable-correct (survive rebuild, unlike an `index.db` cursor).

3. **Durable-tier lock-in of an evolving message vocabulary.** Messages live in irreplaceable `user.db`; a wrong `value_json` shape or addressing scheme is expensive to reshape (copy-forward + consent, no rebuild escape hatch). Overlap awareness that accidentally reads as *blocking* would also train agents to distrust the signal. **Mitigation**: keep the envelope schema (`value_json`) additive and version-tagged; assert `blocking=False` on same-surface overlap in tests; land the scheme behind the two-agent proof (bead 7) as the design validator *before* wide MCP adoption (`ahqd`), so the durable shape is proven under a real round-trip first.

---

**Honesty note**: the coordination *envelope*, degradation ladder, and session-evidence composition are already shipped (`s7ae.1`/`s7ae.4`/`bby.9` ✓) — I did not re-spec them. This spec covers only the open surface: the message bus (blackboard v2), scheduler-mediated advisories, and the two-agent proof. Bead 4 is genuinely blocked on `37t.11` and I flagged it rather than pretending the advisory leg is independently landable.

---

## [a3f79001476bd6c82] SPEC: The Measure Registry (`polylogue-9l5.7` keystone)

I have enough grounding. Here is the implementation-grade spec.

---

# SPEC: The Measure Registry (`polylogue-9l5.7` keystone)

**One-line thesis.** A *measure* is `⟨reducer(column) over unit-frame × grouping × window × comparison × uncertainty⟩` — declared once as a registry row carrying its construct-validity metadata, evaluated through one engine that attaches an uncertainty layer and refuses to emit a bare number when its coverage precondition is unmet. Count becomes the trivial measure (`measure="count"`); every higher analytics layer (9l5.1/.2/.3/.4, .8+) composes through this.

Grounding: `insights/registry.py` (`register()`/descriptor pattern), `insights/archive_models.py` (`SessionLatencyProfilePayload.construct_boundary` is the *only* extant construct-validity-in-payload; generalize it), `insights/rigor.py` (`RigorContract` is the machine-readable per-product rigor precedent), `storage/usage.py:_PROVIDER_USAGE_COVERAGE` (per-origin coverage matrix = the coverage-gate source), `archive/query/expression.py` + `archive/query/metadata.py` (`QueryUnitCountStage`, `QUERY_UNIT_DESCRIPTORS.aggregate_group_fields`, unit sources). Beads: 9l5 (epic), 9l5.7 (keystone), 9l5.2 (cross-provider), 9l5.4 (token economy), fnm/4p1 (one read algebra), o21 (declare-once), fs1.1 (outcome extraction dep).

---

## 1. Schema / DDL, tier, schema-regime

**No durable schema change. No index.db migration.** Measures compute-on-read over columns already materialized in index.db (`session_profiles`, `usage_events`/timeline, cost rollup tables, the `actions` VIEW with `tool_result_is_error`/`exit_code`, `session_links`). This respects CLAUDE.md's "don't reset+reingest the active archive for isolated index additions."

Two code artifacts, zero SQL DDL for v1:

```
polylogue/analytics/stats.py       # pure uncertainty primitives (new package)
polylogue/analytics/measures.py    # MeasureSpec, MEASURE_REGISTRY, evaluate_measure, coverage gate
```

`MeasureSpec` is a frozen pydantic/dataclass declared in code exactly like `InsightType` and `RigorContract` — the registry is a `dict[str, MeasureSpec]`, not a table.

```python
class EvidenceTier(str, Enum):        # ordered weakest→strongest for footnote precedence
    HEURISTIC = "heuristic"; DERIVED = "derived"
    PROVIDER_REPORTED = "provider_reported"; STRUCTURAL = "structural"

class SampleFrame(str, Enum):         # RISK-3 keystone
    LOGICAL_SESSION = "logical_session"   # default: dedup lineage via session_links
    PHYSICAL_SESSION = "physical_session"; ACTION = "action"; MESSAGE = "message"; USAGE_EVENT = "usage_event"

class Reducer(str, Enum):
    COUNT; SUM; MEAN; MEDIAN; QUANTILE; PROPORTION; RATIO; ENTROPY; DISTINCT

class UncertaintyMethod(str, Enum):
    NONE; WILSON; BOOTSTRAP        # WILSON⇒proportions, BOOTSTRAP⇒mean/median/quantile/ratio

@dataclass(frozen=True, slots=True)
class MeasureSpec:
    name: str
    construct: str                       # what it operationalizes (prose)
    unit_label: str                      # display unit ("ratio","tokens","$/session","bits","1/hr")
    unit_frame: SampleFrame              # which rows the reducer folds
    reducer: Reducer
    numerator_expr: str                  # SQL/column expr over the frame (the "formula ref")
    denominator_expr: str | None         # EXPLICIT denominator for RATIO/PROPORTION (never implicit)
    quantile: float | None = None
    default_group_fields: tuple[str, ...] = ()   # ⊆ descriptor.aggregate_group_fields + windows
    evidence_tier: EvidenceTier
    required_coverage: tuple[CoveragePredicate, ...] = ()   # checked at composition (§2.B)
    confounds: tuple[str, ...]           # non-empty is an audit invariant
    uncertainty: UncertaintyMethod
    output_schema: str                   # payload model name for render/openapi
```

**Optional (deferred, not v1):** a *rebuildable* `measure_snapshot` cache table in index.db (schema v25) keyed `(measure, group_key, window)` for scale, dropped-and-recomputed on any index rebuild. Spec it only when §4 scale numbers demand it; keep v1 compute-on-read.

**Schema regime:** derived/index.db → **no migration chain**. But the measure name/reducer/tier vocabularies are embedded in generated surfaces — adding a measure requires regenerating them (the gotcha analog of "new `AssertionKind` breaks `render openapi`"): `devtools render cli-output-schemas openapi query-unit-completions measure-registry devtools-reference && render all --check` (grep `out of sync`).

---

## 2. Core algorithms (pseudocode)

### A. Uncertainty primitives (`analytics/stats.py`) — dependency-lean, scipy behind `[analytics]` extra

```
wilson_interval(k, n, z=1.96):           # ~12 lines, no scipy
    if n == 0: return Interval(nan, nan, degenerate=True)
    p = k/n; d = 1 + z²/n
    center = (p + z²/(2n)) / d
    half   = (z/d) * sqrt(p*(1-p)/n + z²/(4n²))
    return Interval(lo=max(0,center-half), hi=min(1,center+half), point=p, n=n)

bootstrap_ci(values, statistic, iters=2000, z_or_pct=95, rng=seeded):   # RISK: cap iters at scale
    if len(values) < 2: return Interval(statistic(values), degenerate=True)
    boots = [statistic(resample_with_replacement(values, rng)) for _ in range(iters)]
    return Interval(lo=percentile(boots,2.5), hi=percentile(boots,97.5), point=statistic(values), n=len(values))

two_proportion_test(k1,n1,k2,n2): -> z, p_value, diff, wilson_of_each   # pooled-variance normal approx
mann_whitney_u(a, b): -> U, p_value                                     # rank test; latency/cost are non-normal
cliffs_delta(a, b): -> delta ∈ [-1,1]                                   # (P(a>b) − P(a<b)); effect size
shannon_entropy(category_counts): -> bits ∈ [0, log2 k]
histogram_buckets(values, edges): -> counts
```
Core paths use only `wilson_interval`, `bootstrap_ci`, `shannon_entropy` (all hand-rolled). scipy imported lazily for exact tails when the extra is present.

### B. Coverage gate — refuse-to-render (the construct-validity enforcement)

```
check_coverage(spec, group_key) -> CoverageVerdict:
    for pred in spec.required_coverage:
        tier = resolve_tier(pred, group_key)     # from _PROVIDER_USAGE_COVERAGE / timing_provenance / cost_provenance
        if tier is INSUFFICIENT:                  # e.g. ChatGPT origin, estimate_only, for a token measure
            return REFUSED(reason=pred.explain(group_key), footnote=tier_footnote(spec, tier))
    return OK(footnote=tier_footnote(spec, min_tier_seen))   # every OK still carries its tier footnote
```
`CoveragePredicate` examples: `PRICED_PROVENANCE_ONLY` (cost_provenance≠unknown), `TIMESTAMPED_EVENTS_ONLY` (timing_provenance≠sort_key_estimated), `PROVIDER_REPORTED_USAGE` (origin ∈ exact-status set of `_PROVIDER_USAGE_COVERAGE`). A REFUSED verdict renders the footnote and *no number* — never a bare estimate masquerading as a finding (9l5.2).

### C. Measure evaluation (`evaluate_measure`) — the single aggregate path

```
evaluate_measure(spec, filter_spec) -> list[MeasureResult]:
    frame_rows = fetch_unit_rows(spec.unit_frame, filter_spec)     # reuse QUERY_UNIT sql_query_method
    if spec.unit_frame == LOGICAL_SESSION:                          # RISK-3: dedup lineage before folding
        frame_rows = collapse_to_logical(frame_rows, via=session_links)
    groups = group_by(frame_rows, spec.default_group_fields ∪ filter_spec.group_override)
    results = []
    for group_key, rows in groups:
        verdict = check_coverage(spec, group_key)                   # §2.B — gate BEFORE compute
        if verdict.REFUSED: results.append(MeasureResult(group_key, refused=True, footnote=verdict.footnote)); continue
        point, sample = reduce(spec, rows)                          # dispatch on spec.reducer (below)
        interval = attach_uncertainty(spec.uncertainty, point, sample)
        results.append(MeasureResult(group_key, point, interval, n=len(sample),
                                     evidence_tier=spec.evidence_tier, footnote=verdict.footnote,
                                     confounds=spec.confounds))
    return project_origin_payload(results)                          # origin vocab at the boundary

reduce(spec, rows):
    match spec.reducer:
        COUNT       -> (len(rows), rows)
        SUM/MEAN/MEDIAN/QUANTILE -> vals=[eval(spec.numerator_expr,r) for r], (agg(vals), vals)
        PROPORTION  -> k=Σ eval(num,r); n=Σ eval(den,r);  (k/n, (k,n))       # → wilson
        RATIO       -> per-row r_i = num_i/den_i (guard den=0); (mean(r_i), r_i)  # → bootstrap
        ENTROPY     -> (shannon_entropy(counts_by(spec.numerator_expr, rows)), rows)
        DISTINCT    -> (len(set(...)), rows)

attach_uncertainty(WILSON, (k,n))      = wilson_interval(k,n)
attach_uncertainty(BOOTSTRAP, vals)    = bootstrap_ci(vals, statistic=agg_for(spec.reducer))
attach_uncertainty(NONE, point)        = Interval(point, degenerate=True)
```

### D. Comparison axis (two arms)

```
compare_measure(spec, arm_a_filter, arm_b_filter) -> Comparison:
    a = evaluate_measure(spec, arm_a_filter); b = evaluate_measure(spec, arm_b_filter)
    if PROPORTION: return two_proportion_test(a.k,a.n,b.k,b.n)          # + wilson of each arm
    else:          return {u,p = mann_whitney_u(a.sample,b.sample), effect = cliffs_delta(a.sample,b.sample)}
    # antisymmetry invariant: cliffs_delta(a,b) == −cliffs_delta(b,a)  (property test §4)
```

### E. DSL surface (generalize `group by … | count`)

`QueryUnitCountStage` → `QueryUnitMeasureStage(measure: str)` where `count` ≡ `measure="count"`. Grammar in `expression.py`:
```
... | group by <field> | measure <name> [vs <arm-expr>] [with ci]
```
`query_units`/`query_completions` expose the measurable set + each measure's `evidence_tier`+`required_coverage` so agents **discover what is measurable and at what validity before designing an analysis** (the informed-construction affordance, o21).

---

## 3. The 16 named measures (registry rows over existing columns)

| # | name | reducer(numerator / denominator) | frame | tier | uncertainty | key confound / coverage-gate |
|---|---|---|---|---|---|---|
| 1 | `cache_amplification` | RATIO cache_read_tokens / (input+output) | usage_event | provider_reported | bootstrap | cache semantics differ per provider; gate `PROVIDER_REPORTED_USAGE` (9l5.4) |
| 2 | `latency_three_lane_share` | MEAN of per-session thinking/output/tool_ms ÷ wall | logical_session | structural | bootstrap | gate `TIMESTAMPED_EVENTS_ONLY`; agent-lane includes tool time (`construct_boundary`) |
| 3 | `engaged_vs_wall` | RATIO engaged_minutes / wall_minutes | logical_session | derived | bootstrap | `engaged_duration_source` fallback |
| 4 | `tool_mix_entropy` | ENTROPY over `tool` | action | structural | none | low-n groups have unstable entropy |
| 5 | `credit_vs_api_divergence` | RATIO subscription_equivalent_usd / api_equivalent_usd | logical_session | derived | bootstrap | both counterfactual; gate `PRICED_PROVENANCE_ONLY` |
| 6 | `outcome_conditioned_cost` | MEAN total_usd, group by terminal_state | logical_session | derived | bootstrap + compare | `terminal_state_confidence` heuristic (9l5.1) |
| 7 | `pathology_epidemiology` | PROPORTION sessions_with_pathology / sessions | logical_session | heuristic | wilson | detector precision (9l5.3) |
| 8 | `tool_failure_rate` | PROPORTION is_error / total, group by tool | action | provider_reported | wilson | NULL≠error (index v16 keystone) (9l5.2) |
| 9 | `turns_per_task` | MEDIAN authored_user_messages | logical_session | structural | bootstrap | material_origin excludes protocol rows |
| 10 | `cost_per_completed_session` | MEAN total_usd where terminal_state=completed | logical_session | derived | bootstrap + compare | gate `PRICED_PROVENANCE_ONLY` + outcome (fs1.1) |
| 11 | `subagent_usage_rate` | PROPORTION sessions_spawning_subagent / sessions | logical_session | structural | wilson | inheritance=spawned-fresh detection |
| 12 | `babysitting_index` | RATIO authored_user_msgs / engaged_hours | logical_session | derived | bootstrap | engaged fallback (9l5.4) |
| 13 | `context_churn` | RATIO cache_read / unique_input tokens | usage_event | provider_reported | bootstrap | gate `PROVIDER_REPORTED_USAGE` (9l5.4) |
| 14 | `reasoning_output_share` | RATIO reasoning_output / total_output | usage_event | provider_reported | bootstrap | Codex output *includes* reasoning (codex token semantics) |
| 15 | `workflow_shape_distribution` | PROPORTION per workflow_shape | logical_session | heuristic | wilson | `workflow_shape_confidence` |
| 16 | `retry_amplification` | RATIO physical_sessions / logical_sessions | logical_session | structural | bootstrap | lineage dedup correctness (32% dup) |

All numerator/denominator exprs bind to columns that exist today; no reparse, no new materializer.

---

## 4. Migration / rebuild plan

- **Durable tiers untouched** → **no backup manifest gate required** (nothing in `source.db`/`user.db` changes).
- **index.db**: no migration for v1 (compute-on-read). If the deferred `measure_snapshot` cache lands, it is a derived-tier schema bump (v24→v25): edit canonical DDL + rebuild plan, `polylogue ops reset --index && polylogued run` — never an upgrade helper (`devtools lab policy schema-versioning` rejects them).
- **Generated surfaces** (required, same PR as each measure batch): `devtools render cli-output-schemas openapi query-unit-completions measure-registry devtools-reference` then `render all --check` and grep `out of sync`. Topology projection updates because `polylogue/analytics/` is a new package (`devtools render topology-projection topology-status`).
- **MCP**: a `measure`/`aggregate` tool addition → update `EXPECTED_TOOL_NAMES` + tool contract or discovery tests fail.

---

## 5. Test strategy

**Property (Hypothesis, `tests/property`):**
- `wilson_interval(k,n)` ⊆ [0,1]; contains `k/n`; half-width strictly decreasing in `n` for fixed `p`.
- `bootstrap_ci` contains the point statistic; degenerate for n<2.
- `shannon_entropy` ∈ `[0, log2 k]`; maximal iff uniform; 0 iff single category.
- PROPORTION reducer == `k/n`; RATIO guards `den=0` (never raises).
- `cliffs_delta(a,b) == −cliffs_delta(b,a)` (comparison antisymmetry).

**Metamorphic:**
- **Partition invariance:** measure over `frame = A ⊎ B` equals the coverage-weighted combine of the two sub-frames (sums add; proportions pool k,n; entropy recomputes from merged counts).
- **Scale invariance:** multiply every group's row multiplicity by c → PROPORTION/ENTROPY unchanged, SUM scales by c, MEAN/MEDIAN unchanged.
- **Coverage monotonicity:** appending a row from a coverage-excluded origin (ChatGPT `estimate_only`) does **not** change a `PROVIDER_REPORTED_USAGE`-gated measure's value.
- **Refusal, not fabrication:** on a seeded corpus with a partial-provenance origin, a gated measure for that group renders `refused=True` + footnote and emits **no number** (asserts the 9l5.2 honesty contract).

**Scale (`tests/benchmarks`, guarded off default):** `evaluate_measure` over the 16K-physical-session frame collapses to the ~8.8K logical frame (assert `logical_n < physical_n` — RISK-3) and completes within a bounded budget; bootstrap iters capped (2000) so the 16-measure sweep stays interactive.

**Rigor-audit extension (`insights/rigor.py`/`audit.py`):** `insight_rigor_audit` gains a measure lane — asserts every `MeasureSpec` has non-empty `construct`, a resolvable `evidence_tier`, non-empty `confounds`, and (for RATIO/PROPORTION) an explicit `denominator_expr`; and that every measure output on CLI/MCP/API carries its tier footnote. A measure emitting a number without a footnote fails the audit.

---

## 6. Bead breakdown (children of 9l5.7)

1. **`analytics/stats.py` uncertainty substrate** — *AC: `wilson_interval`, `bootstrap_ci`, `two_proportion_test`, `mann_whitney_u`, `cliffs_delta`, `shannon_entropy`, `histogram_buckets` are pure, property-tested, and importable with no scipy (scipy behind the `[analytics]` extra).*
2. **`MeasureSpec` + `MEASURE_REGISTRY` declare-once scaffold** — *AC: the registry holds ≥16 specs each declaring construct/formula/denominator/evidence_tier/required_coverage/confounds/uncertainty/sample_frame; `render measure-registry` regenerates its doc and `render all --check` passes.*
3. **`evaluate_measure` reducer engine** — *AC: one engine dispatches count/sum/mean/median/quantile/proportion/ratio/entropy, attaches the declared uncertainty layer, and returns `MeasureResult(point, interval, n, tier, confounds)`; `count` routes through it as `measure="count"`.*
4. **Coverage-gate composition** — *AC: a measure whose `required_coverage` is unmet on a group renders `refused=True` with a tier footnote sourced from `_PROVIDER_USAGE_COVERAGE`/`timing_provenance`, never a bare number; every OK result still carries its tier footnote.*
5. **16 named measures as registry rows over existing columns** — *AC: all 16 in §3 evaluate on the seeded corpus, each with the correct tier footnote and (where gated) a REFUSED verdict on the partial-provenance origin.*
6. **DSL `measure` stage (generalize `count`)** — *AC: `<unit> where … | group by <field> | measure <name>` parses and executes; `query_units`/`query_completions` expose measurable measures with their evidence_tier and required_coverage.*
7. **Comparison axis** — *AC: `measure <name> … vs <arm>` returns effect size + rank/proportion test with per-arm CIs, sign-antisymmetric under arm swap.*
8. **Rigor-audit + automatic surface footnotes** — *AC: `insight_rigor_audit` audits `MEASURE_REGISTRY` (non-empty construct/confounds, resolvable tier, explicit denominator for ratios) and fails if any measure output on CLI/MCP/API lacks its tier footnote.*

(Beads 1–2 and 3–4 may pair into single PRs; 5 is one PR per measure-family batch, not per measure.)

---

## 7. Top-3 risks

1. **Coverage-gate bypass = silent construct-invalid number.** If any surface emits an aggregate off a path other than `evaluate_measure` (e.g. a legacy `group by … | count`), it escapes the gate and the tier footnote — exactly the "partial provenance masquerading as a finding" failure 9l5.2 exists to kill. *Mitigation:* make `count` a measure so there is **one** aggregate path; add an audit/lint (bead 8) that fails if any rendered aggregate lacks a `MeasureResult` tier tag.
2. **Non-injective origin projection in per-origin grouping.** `GEMINI` **and** `DRIVE` both collapse to `AISTUDIO_DRIVE`; grouping on the raw `source_name` vs the projected `Origin` splits or merges arms wrongly, corrupting cross-origin comparisons. *Mitigation:* group on canonical `Origin` and project at the boundary via the existing `project_origin_payload`; metamorphic test asserts a cross-origin measure keeps one coverage tier **per origin** and does not merge Gemini/Drive silently.
3. **iid assumption violated by lineage duplication.** Forks/resumes/auto-compaction physically replay parent prefixes (~16K physical vs ~8.8K logical, ~32% duplicate messages). Treating physical sessions as independent inflates `n`, falsely shrinks Wilson/bootstrap intervals, and double-counts sums. *Mitigation:* `SampleFrame.LOGICAL_SESSION` is the **default** frame — measures collapse to logical sessions via `session_links` before folding; `PHYSICAL_SESSION` must be opted into per-spec; scale test asserts `logical_n < physical_n`. (`retry_amplification` is the one measure that legitimately reads the physical/logical ratio.)

*(Secondary: scipy in a core path — mitigated by hand-rolled Wilson/bootstrap/entropy, scipy strictly optional behind `[analytics]`.)*

---

## [ad6d451431af3bad5] Wave-5 Spec: Annotation-Recipe Substrate

I have thorough grounding. Here is the implementation spec.

---

# Wave-5 Spec: Annotation-Recipe Substrate

Reusable, parameterized recipes that instruct an agent (or deterministically prefill) to annotate sessions/messages into queryable, recursive-safe candidate assertions. Grounded in `user_write.py`, the assertions DDL (`user.py`), the three insight readers, and beads 37t / 37t.1 / 37t.2 / 37t.12 / pj8.

## 0. What already exists (verified, do not rebuild)

- **Assertions substrate is complete.** `assertions` table (`user.py:12`, `USER_SCHEMA_VERSION=4`, STRICT) has `kind`/`scope_ref`/`target_ref`/`key`/`value_json`/`body_text`/`author_ref`/`author_kind`/`evidence_refs_json`/`status`/`visibility`/`confidence`/`staleness_json`/`context_policy_json`/`supersedes_json`. `kind` is schema-free `TEXT` (no CHECK) — new kinds cost no SQL migration.
- **Candidate→judge lifecycle exists.** `upsert_assertion` (`user_write.py:901`), `judge_assertion_candidate` (`:1245`, decisions accept/reject/defer/supersede → JUDGMENT row + promoted ACTIVE assertion via `_promote_candidate_assertion`), `list_assertion_candidates`/`list_assertion_candidate_reviews`, `mark_assertion_status`. `ASSERTION_CLAIM_KINDS` = decision/caveat/blocker/lesson/run_state/transform_candidate/pathology.
- **Two derived prefill writers already mirror insights → CANDIDATE assertions:** `upsert_transform_candidate_assertions` (from `SessionDigest.decision_candidates`, author_kind=`transform`) and `upsert_pathology_findings_as_assertions` (from `PathologyFinding`, author_kind=`detector`). Both set `status=CANDIDATE`, `context_policy={"inject":False,"promotion_required":True}`, deterministic ids, and refuse to downgrade an already-judged row. **These are the prototype for every recipe skeleton.**
- **Insight readers the postmortem recipe consumes:** `extract_phases` → `SessionPhase(message_range, tool_counts, word_count, duration_ms, start/end_time)` (`archive/phase/extraction.py`); `detect_session_pathologies` → `PathologyFinding(kind, severity, detail, occurrence_count, evidence_refs, detector_version)` (`insights/pathology.py:191`); `SessionDigest.decision_candidates`/`SessionDigestEvent` with `raw_refs` (`insights/transforms.py`).
- **MCP prompt channel:** `@mcp.prompt()` decorator, `register_prompts` (`mcp/server_prompts.py:219`) — prompts return prose with prefilled context. This is where the recipe's PROMPT surface plugs in (aligns with pj8).
- **CLI verbs are a closed frozenset** (`cli/verb_names.py`: analyze/continue/delete/mark/read/select); root commands register via `register_root_commands` (`cli/click_app.py:426`).

## 1. Schema / DDL + tier

**No durable SQL migration is required.** The recipe substrate is program logic + prose; its *output* is ordinary assertion rows.

- **Recipe library = in-repo YAML, a generated/checked surface** (recommendation: YAML over DB-object). Rationale: a recipe is code (prompt text + selector + skeleton logic), belongs in git under review, and should be render-`--check`-validated like `INSIGHT_REGISTRY`/topology. Storing built-in recipes in `user.db` would put program logic in a durable, irreplaceable, backup-gated tier — wrong durability axis. Location: `polylogue/recipes/*.yaml` loaded into typed `Recipe` descriptors (Pydantic), registered in a `RECIPE_REGISTRY` (mirrors `insights/registry.py`).
- **Operator-authored recipes (optional, later) = `AssertionKind.RECIPE`** — a new enum member only. Because `kind` is TEXT/no-CHECK, this needs **zero SQL migration**: just the enum + render regen (`render openapi`, `render cli-output-schemas`) + a `user_audit` surface entry (the every-kind-has-a-surface invariant). Keep out of scope for the first phase; note it.
- **Recipe-run provenance rides existing columns** (no new columns): each emitted assertion gets `scope_ref = "recipe:{name}@v{version}"`, `author_kind = "recipe"`, `author_ref = scope_ref`, deterministic `assertion_id`. Query-back by recipe = `list_assertion_claims(scope_ref="recipe:postmortem@v1")` (already supported).
- **Optional run-ledger in `ops.db` (disposable tier, DDL edited freely):** `recipe_runs(run_id, recipe_name, recipe_version, params_json, target_ref, emitted_count, created_at_ms)` for observability. Disposable → no migration ceremony. Include only if a run-history surface is wanted.

**`Recipe` descriptor shape (YAML → Pydantic):**
```
Recipe:
  name: str                      # "postmortem"
  version: int
  description: str
  params: [ParamSpec{name,type,required,default}]      # -> generated Pydantic param model
  selector: str                  # parameterized DSL expr, e.g. "session:{session_id}" / "repo:{repo} since:{since}"
  prompt_template: str           # agent instruction; {param} + {insight_context} interpolation
  skeletons: [SkeletonSpec]      # deterministic pre-fills (below)
SkeletonSpec:
  kind: AssertionKind            # decision|blocker|caveat|lesson|note|...
  source: enum{pathology, decision_candidate, phase, static}
  body_template: str             # {finding.detail} / {candidate.text} / free
  key_template: str | null
  value_template: dict | null
  evidence_from: enum{finding.evidence_refs, candidate.raw_refs, phase.message_range, none}
  # status/context_policy are NOT author-settable — forced by the recursive-safety gate
```

## 2. Recipe execution algorithms (pseudocode)

**A. Recursive-safety gate (the load-bearing invariant).** Today `upsert_assertion` defaults `author_kind='user', status=ACTIVE`; the derived writers set CANDIDATE *manually*, and `blackboard_post` lets an agent write `author_kind='agent'` that lands **ACTIVE** (gap — see §Audit). Introduce one chokepoint:
```
def coerce_agent_authored(author_kind, status, context_policy):
    if author_kind in {"user"}:                     # only the operator is trusted-active
        return status, context_policy
    # any non-user author (agent, recipe, detector, transform, subagent)
    if status in TERMINAL_JUDGED {accepted,rejected,deferred,superseded,deleted}:
        return status, context_policy               # never resurrect a judged row
    forced = CANDIDATE
    cp = {**(context_policy or {}), "inject": False, "promotion_required": True}
    return forced, cp
```
Call it inside `upsert_assertion` (single enforcement point) so the transform/pathology writers, MCP agent writes, and recipes all inherit it. Promotion to ACTIVE happens *only* through `judge_assertion_candidate` (author_kind=user). This is the "QUOTED→OPERATOR transition" 37t.11 depends on.

**B. `run_recipe(name, params, mode)`:**
```
recipe = RECIPE_REGISTRY[name]
p = recipe.param_model(**params)                    # typed validation, defaults
targets = resolve_selector(recipe.selector.format(**p))   # DSL -> session ids
emitted = []
for session_id in targets:
    ctx = gather_insight_context(session_id)        # C below
    for skel in recipe.skeletons:
        rows = expand_skeleton(skel, session_id, ctx, p)   # D below
        for r in rows:
            aid = deterministic_id("recipe", name, version, session_id, skel.kind, r.key, *r.evidence)
            existing = read_assertion_envelope(conn, aid)
            if existing and existing.status != CANDIDATE:   # idempotent, judged-safe
                emitted.append(existing); continue
            emitted.append(upsert_assertion(
                assertion_id=aid, scope_ref=f"recipe:{name}@v{version}",
                target_ref=f"session:{session_id}", kind=skel.kind, key=r.key,
                value=r.value, body_text=r.body, author_ref=scope_ref,
                author_kind="recipe", evidence_refs=r.evidence))
                # status/context_policy forced to CANDIDATE by gate (A)
if mode == "prompt":                                # agent-facing
    return render_prompt(recipe.prompt_template, p, ctx, emitted_refs=[a.id for a in emitted])
return emitted                                      # prefill mode: candidate rows only
```
Two modes, one code path: **prefill** deterministically lands skeleton candidates; **prompt** additionally emits an instruction that references those candidates by `assertion:{id}` and asks the agent to complete freeform fields (e.g. the LESSON body) via MCP writes — which themselves land as new candidates. The operator judges the union via 37t.12.

**C. `gather_insight_context(session_id)`** — the postmortem prefill source:
```
projection = build_run_projection(session_id)
return {
  "pathologies": detect_session_pathologies(projection),      # PathologyFinding[]
  "decisions":   build_session_digest(session_id).decision_candidates,  # DecisionCandidate[]
  "phases":      extract_phases(load_session(session_id)),    # SessionPhase[]
}
```

**D. `expand_skeleton` per source (postmortem recipe):**
```
match skel.source:
  pathology:          for f in ctx.pathologies:
                          yield Row(kind=BLOCKER if f.severity>=high else CAVEAT,
                                    key=f.kind, body=f.detail,
                                    value={severity,occurrence_count,detector_version},
                                    evidence=[ref.format() for ref in f.evidence_refs])
  decision_candidate: for c in ctx.decisions:
                          yield Row(kind=DECISION, key=c.kind, body=c.text,
                                    evidence=[r.to_evidence_ref().format() for r in c.raw_refs])
  phase:              for ph in ctx.phases:                    # optional structural note
                          yield Row(kind=NOTE, body=phase_summary(ph),
                                    value={message_range,tool_counts,duration_ms}, evidence=[])
  static:             yield Row(kind=LESSON, body="", ...)     # empty skeleton for agent to fill
```
This *is* the existing pathology/transform mirroring generalized into a declarative table — the postmortem recipe subsumes `upsert_pathology_findings_as_assertions` + `upsert_transform_candidate_assertions` as two of its skeleton sources (keep those functions; have the recipe call them or share their id helpers to preserve idempotent ids).

## 3. Migration

- **Durable SQL migration: none.** `assertions`/`user.db` DDL unchanged; `USER_SCHEMA_VERSION` stays 4. Recipe output uses existing columns.
- **If/when `AssertionKind.RECIPE` (operator recipes) is added:** enum member only + `render openapi` + `render cli-output-schemas` regen (enum is embedded in both) + a `user_audit` surface entry (else the every-kind invariant fails) + a scope_ref/author_ref using a *registered* ObjectRef kind (`recipe:`). No `PRAGMA user_version` bump (schema-free TEXT). This matches the #2383 pathology precedent exactly.
- **Ops-tier `recipe_runs` (if adopted):** disposable tier — add DDL, no numbered migration; a schema mismatch just resets ops.db.
- **Topology projection:** adding `polylogue/recipes/` module(s) requires `devtools render topology-projection && render topology-status` (else `render all --check` fails).
- **New MCP tools / CLI verbs:** update `EXPECTED_TOOL_NAMES` + tool contracts; new root command needs `render cli-reference` / help-snapshot regen.

## 4. Test strategy

- **Recursive-safety property (highest value):** for every `author_kind ∈ {agent, recipe, detector, transform, subagent}` and every non-terminal input status, `upsert_assertion` yields `status=CANDIDATE` and `context_policy.inject=False, promotion_required=True`; `author_kind=user` is unchanged. Hypothesis over author_kind × status.
- **Judged-row non-resurrection:** re-running a recipe over a session whose candidate was `accepted`/`rejected` does not downgrade it (mirrors existing `existing.status != CANDIDATE` guard tests).
- **Prefill determinism / idempotency:** same seeded session → byte-identical `assertion_id` set on re-run; no duplicate rows. Reuse `corpus_seeded_db` + `frozen_clock`.
- **Postmortem correctness:** seed a session with a known wasted-loop pathology + one decision candidate → assert exactly one BLOCKER/CAVEAT skeleton (evidence_refs == finding's) and one DECISION skeleton (evidence_refs == candidate's raw refs), each resolvable via `resolve_ref`.
- **Round-trip (ties into 37t.12):** run recipe → candidate → `judge_assertion_candidate(accept)` → new ACTIVE assertion appears in `list_assertion_claims(statuses=[active])`, queryable by `scope_ref`.
- **Recipe descriptor validation lane:** every YAML loads into a `Recipe`; param models generate; unknown `kind`/`source`/`evidence_from` rejected — a `devtools lab` check + `render all --check` gate.
- **MCP declare/annotate write gate:** an agent MCP write lands CANDIDATE; assert an agent *cannot* produce ACTIVE (negative test) — this also fixes the `blackboard_post` ACTIVE gap.
- **CLI:** `recipe list` / `recipe show` snapshot (syrupy); `recipe run --param` bad-param → UsageError; help-snapshot regen.

## 5. Bead breakdown (children under 37t; recipe substrate is adjacent to 37t.1/37t.2/pj8)

1. **Recursive-safety gate in the assertion chokepoint** (feature; prerequisite for all). AC: `coerce_agent_authored` enforced inside `upsert_assertion`; all non-user authors → CANDIDATE + inject:false + promotion_required; user unchanged; terminal-judged rows never resurrected; `blackboard_post` agent write now lands CANDIDATE not ACTIVE; Hypothesis property test green; existing transform/pathology writers still pass.
2. **`Recipe` descriptor + `RECIPE_REGISTRY` + YAML loader** (feature). AC: typed `Recipe`/`ParamSpec`/`SkeletonSpec` Pydantic models; `polylogue/recipes/*.yaml` loaded + validated; a `devtools lab` / `render --check` lane rejects malformed recipes; topology projection regenerated.
3. **`run_recipe` engine (prefill + prompt modes)** (feature). AC: deterministic idempotent skeleton emission with `scope_ref=recipe:{name}@v{version}`, author_kind=recipe; re-run over unchanged session emits zero new rows; prompt mode returns instruction referencing emitted candidate refs.
4. **Built-in `postmortem` recipe** (feature). AC: consumes `detect_session_pathologies` + `SessionDigest.decision_candidates` + `extract_phases`; emits BLOCKER/CAVEAT/DECISION/NOTE skeletons + empty LESSON with correct evidence_refs; shares deterministic-id helpers with the existing pathology/transform mirrors (no id drift); seeded-session correctness test.
5. **CLI surface: `polylogue recipe {list,show,run}`** (task). AC: root command group (not a query verb — avoids the closed-verb-set/positional-shift gotcha); `run <name> --param k=v` validates params, prints emitted candidate refs; help + cli-reference snapshots regenerated. (Note the literal `run-recipe` ask is satisfied as `recipe run`; document the rename rationale.)
6. **MCP `run_recipe` prompt + generic `declare_assertion`/`annotate` write tool** (feature; closes the claim-write gap in §Audit). AC: `@mcp.prompt()` `run_recipe`/`postmortem_last` registered (pj8-aligned); a write tool lets an agent declare DECISION/CAVEAT/LESSON/BLOCKER/HANDOFF candidates with target+body+evidence, landing CANDIDATE via the gate; `EXPECTED_TOOL_NAMES` + tool contracts updated; discovery test green.
7. **Assertions as a DSL query-back unit source** (feature; enables `find "kind:lesson repo:x"`). AC: assertion claims become a queryable unit in the query grammar (or a documented MCP-only query-back path if DSL integration is deferred); `list_assertion_claims` gains scope_ref/repo filters if needed. *Scope-flag: largest item; can be deferred to a follow-up if the DSL lowering is heavy — mark explicitly.*
8. **(Optional) Recipe-run ledger in ops.db + `AssertionKind.RECIPE` for operator recipes** (task). AC: disposable `recipe_runs` table; enum+render+user_audit surface for operator-authored YAML-equivalent recipes stored in user.db.

Depends-on: 1 → {2,6}; 2 → 3 → 4 → 5; 6 needs 1; 7 independent-ish; couples to 37t.12 (judgment) and pj8 (prompt discoverability).

## 6. Top-3 risks

1. **Recursive-safety enforcement placement.** If the gate lives only in the recipe path (not in `upsert_assertion`), agent MCP writes (`blackboard_post` already does this) keep landing ACTIVE and self-injecting — defeating the whole point and re-opening the QUOTED→OPERATOR hole 37t.11 flags as blocking-security. Must be one chokepoint, with an allowlist for `user` only. Verify no legitimate user path is accidentally forced to candidate (marks/tags/annotations authored by the operator via CLI must stay ACTIVE — they carry author_kind=user).
2. **Query-back is the real completeness gap, and it's the biggest item.** Today assertions are queryable only via MCP `list_*` tools with target/kind/status filters — the `find` DSL indexes the *index tier*, not `user.db`. "Annotate + query back" is only half-true until bead 7 lands. Risk of shipping a write-heavy substrate whose outputs can't be found by the DSL the CLI/agents actually use. Scope honestly: phase-1 = MCP `list_assertion_claims(scope_ref=...)` query-back; DSL unit-source is a named follow-up, not silently claimed done.
3. **Skeleton id / provenance drift vs. the two existing mirrors.** `upsert_pathology_findings_as_assertions` and `upsert_transform_candidate_assertions` already write PATHOLOGY/TRANSFORM_CANDIDATE candidates with their own deterministic ids. If the postmortem recipe re-emits the same findings under new recipe-scoped ids, the same finding becomes two candidate rows (double judgment burden, inconsistent evidence). Mitigation: the recipe must reuse `assertion_id_for_pathology_finding` / `assertion_id_for_transform_candidate` (share helpers), and treat those functions as the recipe's skeleton emitters rather than a parallel path — otherwise idempotency and the judged-row guard operate on divergent id spaces.

---

**Key files:** `polylogue/storage/sqlite/archive_tiers/user_write.py` (assertion writers, judge lifecycle, deterministic ids), `.../user.py` (DDL, `USER_SCHEMA_VERSION=4`), `polylogue/core/enums.py:399` (AssertionKind/Status/Visibility), `polylogue/insights/pathology.py`, `polylogue/insights/transforms.py`, `polylogue/archive/phase/extraction.py` (the three prefill sources), `polylogue/mcp/server_prompts.py:219` + `server_mutation_tools.py:150` (prompt + agent-write surfaces; note the `blackboard_post` ACTIVE gap), `polylogue/cli/verb_names.py` + `click_app.py:426` (verb vs root-command boundary). Beads: **37t.1** (claim-kind first-writers), **37t.2** (inline marker protocol — an alternate ingestion path into the same candidates), **37t.12** (judgment queue — the promotion gate this substrate feeds), **pj8** (MCP prompt discoverability — where recipe prompts register).

---

## [aee0f913564c1db30] Wave-5 Spec — Steady-State Observability of Polylogue Itself

I have everything I need — the full metrics/health surfaces, the ops.db DDL, the SLO-ladder pattern (cursor_lag_alert), the bootstrap/versioning semantics, and the status CLI. Here is the spec.

---

# Wave-5 Spec — Steady-State Observability of Polylogue Itself

Grounded in `polylogue/daemon/metrics.py` (1858 L), `polylogue/daemon/health.py` (1226 L), `polylogue/daemon/cursor_lag_alert.py`, `polylogue/storage/sqlite/archive_tiers/ops.py` (OPS_DDL, `OPS_SCHEMA_VERSION=1`), `bootstrap.py`, `ops_write.py`, `cli/commands/status.py`.

## 0. Gap analysis (what exists vs what steady-state needs)

The current surfaces measure **point-in-time state**, not **rates/trends/objectives**:

| Have today | Missing for steady-state |
|---|---|
| `_recent_attempt_durations` → min/mean/max of last 50 `convergence_time_s` | true **capture→durable p50/p95/p99**, live-tail scoped |
| `_convergence_debt_by_stage` → level gauge | **slope** dB/dt (growing before threshold) |
| `_embedding_state` → pending count + latest run | **burn-down ETA** = pending / drain-rate |
| `fts_freshness_ready` → boolean per surface | **drift magnitude** (missing+excess+dup rows) + trend |
| `check_health` → worst-severity ladder | **idle-healthy vs stalled** verdict; **error-budget burn rate** |
| `cursor_lag` static+anomaly (stuck-vs-idle already solved per-family) | that distinction **lifted to a top-level runtime verdict** |
| `daemon_events` table exists | **no writer**; unused — ideal meta-session substrate |
| `/metrics`, `/healthz/{live,ready}` | **`status --health` one-glance**, **`/dashboard`**, **/metrics SLO contract test** |

The root structural gap: **metrics are pull-computed on scrape**, so there is no persisted time series inside polylogue to compute slope/ETA/burn-rate without an external TSDB. `cursor_lag_samples` already proves the fix (periodic persisted samples + retention GC). The spec generalizes that into one measure-sample substrate.

---

## 1. SLO / schema surface

### 1a. Measure algebra (the convergent design)

One homogeneous **sample stream**, measure-typed and unit-typed, with a small closed set of reducers. All SLOs plug into one evaluation kernel instead of bespoke per-metric SQL.

**Measures** (construct-valid definitions):

- `ingest_latency_seconds` — *per-event*. `(durable_at_ms − capture_at_ms)/1000` for each completed **live-tail** attempt. `capture_at_ms` = min source-file mtime / `raw_sessions.acquired_at_ms` of records in the attempt (NOT message-send time; NOT bulk-import age). **Scoped to live origins** `{claude-code-session, codex-session, gemini-cli-session}`; bulk `{chatgpt-export, claude-ai-export, …}` excluded by construction (their "latency" is historical, not a defect). Labels: `origin`, `storage_route`.
- `convergence_lag_seconds` — *gauge*. Freshness age of the freshest-required derived model = `max(` oldest `raw_sessions` with no matching `sessions` row; oldest unresolved `convergence_debt.created_at_ms`; oldest `sessions` row without a `session_profile` `)`, all as `now − t`. This is "how stale is what should be materialized."
- `embedding_backlog_count` — *gauge*. `pending_sessions` from `_embedding_state`.
- `convergence_debt_count` — *gauge*. Unresolved `convergence_debt` rows (existing `_convergence_debt_by_stage` summed).
- `fts_drift_rows` — *gauge*. `Σ_surface (missing_rows + excess_rows + duplicate_rows)` from `fts_invariant_snapshot_sync` (today only boolean `ready` is exposed).

**Reducers** (pure functions over a windowed sample slice):
- `level(m)` — latest gauge value.
- `quantile(m, q, W)` — p50/p95/p99 over per-event samples in window `W`.
- `slope(m, W)` — least-squares dvalue/dt (units/s). **Leading indicator**: `slope>0` on a backlog measure = growing *before* it crosses an absolute threshold.
- `eta(m, W)` — `level / max(−slope, ε)` when `slope<0` (draining); **undefined/−1 when `slope≥0`** (growing or flat → never drains).
- `burn(slo, W)` — `violation_fraction(W) / (1 − objective)`. Multi-window multi-burn (Google SRE): fast `(1h, 14.4×)` → page/CRITICAL; slow `(6h, 6×)` → ticket/ERROR.

### 1b. SLO objectives (config `[health.slo]`, mirroring `[health.cursor_lag]`)

```toml
[health.slo]
ingest_latency_p95_target_s   = 30
ingest_latency_p99_target_s   = 120
convergence_lag_target_s      = 300
embedding_backlog_eta_warn_s  = 3600     # ETA beyond this → WARNING
fts_drift_rows_error          = 1        # any drift row → ERROR
objective_ratio               = 0.99     # 99% of samples within target over window
window_fast_s                 = 3600
window_slow_s                 = 21600
dedup_window_s                = 3600
[health.slo.families.claude-code-session]  # per-origin overrides, same shape as cursor_lag
ingest_latency_p95_target_s   = 10
```

Loaded via a `SloThresholds` dataclass following `CursorLagThresholds.for_family` (monotonic clamp, per-family override → default). Evaluation reuses the `cursor_lag_alert` dedup/escalation contract verbatim: escalations and resolutions fire immediately even inside the dedup window; a single `severity=ok` resolution alert on recovery.

### 1c. ops.db schema (`OPS_SCHEMA_VERSION 1 → 2`)

**One new table** (keep the migration tight):

```sql
CREATE TABLE IF NOT EXISTS slo_samples (
    sample_id     TEXT PRIMARY KEY,
    measure       TEXT NOT NULL,                 -- closed set (5 measures above)
    labels_json   TEXT NOT NULL DEFAULT '{}',    -- {origin,route,surface,stage}
    value         REAL NOT NULL,
    unit          TEXT NOT NULL,                 -- 'seconds' | 'count' | 'rows'
    sampled_at_ms INTEGER NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS idx_slo_samples_measure_time
ON slo_samples(measure, sampled_at_ms DESC);
```

Holds **both** per-event latency samples (one row per completed live attempt) and periodic gauge samples — the unified stream the reducers consume. No duplication of `ingest_attempts` (that stays the attempt event log).

**Reuse `daemon_events`** (already in DDL, no writer): add `record_daemon_event(conn, kind, operation_id, payload)` in `ops_write.py`. SLO breaches, verdict transitions, attempt lifecycle, GC → `daemon_events` rows (`kind ∈ {slo_breach, verdict_change, ingest_attempt, catchup_run, gc}`). This doubles as the **meta-session** substrate, so no separate breach table is needed (breach dedup stays in-memory, matching cursor_lag).

### 1d. New Prometheus series (closed-set labels only)

```
polylogue_ingest_latency_seconds{quantile,origin}   gauge   # p50/p95/p99
polylogue_convergence_lag_seconds                   gauge
polylogue_backlog_slope{measure}                    gauge   # units/s, sign = leading indicator
polylogue_embedding_backlog_eta_seconds             gauge   # -1 when not draining
polylogue_fts_drift_rows{surface}                   gauge
polylogue_slo_burn_rate{slo,window}                 gauge   # window=fast|slow
polylogue_slo_objective_ratio{slo}                  gauge   # attained ratio over window
polylogue_runtime_verdict{verdict}                  gauge   # one-hot, closed set
```

Wire into `format_metrics` after `_emit_embedding_metrics`, sourced from `slo_samples` via reducers; degrade to zero-sample skeleton (existing `_emit_metric([])` idiom) when `slo_samples` absent.

---

## 2. Algorithms (pseudocode)

**Sampler** (runs in the MEDIUM health cadence, sole writer, bounded, GC'd like `gc_cursor_lag_samples`):
```
def sample_slo_measures(ops_conn, index_db, embeddings_db, now_ms):
    for m, val, unit, labels in [
        latency_samples_for_recently_completed_attempts(),   # per-event, live-tail only
        ("convergence_lag_seconds", freshness_age(index_db, now_ms), "seconds", {}),
        ("embedding_backlog_count", pending_sessions(embeddings_db), "count", {}),
        ("convergence_debt_count",  unresolved_debt(ops_conn), "count", {}),
        ("fts_drift_rows",          fts_drift_magnitude(index_db), "rows", {}) ]:
        record_slo_sample(ops_conn, m, labels, val, unit, now_ms)
    gc_slo_samples(ops_conn, retention_days = max(retention, 2*slow_window_days))
```

**Reducers** (pure, `frozen_clock` in tests):
```
def quantile(samples, q):          # nearest-rank on sorted values; [] -> None
def slope(samples, W):             # least squares on (t-t0, v); <2 pts -> 0.0
def eta(level, slope_val):         # slope<0: level/-slope ; else: -1 (undefined)
def burn(samples, target, cmp, objective, W):
    viol = fraction(samples in W where cmp(v, target) violated)
    return viol / (1 - objective)          # >1 => burning budget faster than allowed
```

**Multi-window burn alert** (mirrors `evaluate_cursor_lag`):
```
for slo in slos:
    fast = burn(win_fast); slow = burn(win_slow)
    sev = CRITICAL if fast >= 14.4 and slow >= 1 else \
          ERROR    if slow >= 6 else \
          WARNING  if slow >= 3 else OK
    emit_with_dedup(slo, sev, state)        # escalate/resolve immediately
```

**Runtime verdict — idle-vs-stalled keystone** (construct validity: a non-zero backlog is a defect *only* when not draining AND work is offered):
```
offered = running_attempts > 0 or stuck_cursor_files > 0 or (backlog>0 and slope>=0)
if any_slo_fast_burn:                      verdict = CRITICAL   # budget torched
elif backlog == 0 and running_attempts==0: verdict = HEALTHY_IDLE   # caught up, quiet
elif backlog > 0 and slope < 0:            verdict = CONVERGING     # draining, eta finite
elif backlog > 0 and slope >= 0 and running_attempts==0 and stuck_cursor_files>0:
                                           verdict = STALLED        # pending, not advancing (ERROR)
elif running_attempts > 0 and conv_lag <= target: verdict = BUSY    # OK
else:                                      verdict = DEGRADED
```
This lifts `cursor_lag`'s per-family stuck-vs-idle into one archive-wide verdict spanning all five backlog measures — the disambiguation the spec calls for.

**Meta-session projection** (synthetic, read-through — never written to index.db):
```
def runtime_meta_session(ops_db, boot_id):
    sid = "polylogue-runtime:" + boot_id
    msgs = merge_ordered(
        daemon_events(kind, payload, ts),
        ingest_attempts(lifecycle),
        slo_breaches)                       # each -> a message
    for msg: material_origin = RUNTIME_PROTOCOL   # excluded from authored/cost
    return Session(origin="polylogue-runtime", messages=msgs, blocks=stage_payloads)
```
Surfaced through the existing read path (`read --view transcript`, MCP `get_session_tree`) as a virtual origin the reader recognizes — the archive observing itself with its own query tools.

---

## 3. Migration

ops.db is the **disposable derived tier** (`bootstrap.py:ARCHIVE_TIER_SPECS[OPS].backup_required=False`, not in `DURABLE_MIGRATION_TIERS`). Per repo policy (`docs/internals.md`, `devtools lab policy schema-versioning`): **no migration chain — edit canonical DDL + rebuild.**

Mechanics (verified in `initialize_archive_database`): version match → early return; existing ops.db at v1 with new `OPS_SCHEMA_VERSION=2` → raises *"move it aside and rebuild the archive root."* The `_ensure_ops_*` additive-column helpers run **only** on fresh init, so they do NOT auto-apply to an existing matched DB.

**Plan:** add `slo_samples` to `OPS_DDL`, bump `OPS_SCHEMA_VERSION → 2`, ship a rebuild plan: `polylogue ops reset --ops && polylogued run`. Reset re-tails sources from scratch (idempotent by content hash — no data loss to index/source/user tiers; skipped-write on matching hash). Cost: lost ingest cursors → one full re-scan; lost `cursor_lag_samples` + `slo_samples` → anomaly baseline + slope history cold-start (re-accumulate over `baseline_window_days`). **Therefore land ALL Wave-5 ops.db additions in the single v2 bump** (operator doctrine: don't repeatedly reset+reingest for isolated additions). `daemon_events` writer needs no schema change (table already present).

---

## 4. Test strategy

- **Reducer unit tests** (`frozen_clock`): quantile edge cases (empty→None, single, ties, nearest-rank); slope sign & zero on <2 points; **eta undefined (−1) when slope≥0** (growing backlog never "drains"); burn-rate math at 3×/6×/14.4× boundaries.
- **Construct-validity guards** (the load-bearing tests): bulk-export attempt contributes **zero** `ingest_latency_seconds` samples; `backlog==0 & idle → HEALTHY_IDLE` (not STALLED); `backlog>0 & slope≥0 & stuck → STALLED/ERROR`; `convergence_lag_seconds` = max over the three derived-model lags on a fixture with a stale-profile session.
- **Degradation/migration**: fresh ops.db (no `slo_samples`) → metrics skeleton still emits all series at zero; missing tables never raise (existing `_table_exists` idiom); v2 rebuild produces `slo_samples`.
- **/metrics contract test** (extends existing): full stable NAME set present incl. the 8 new series; every series has `# HELP`/`# TYPE`; exposition parses under a strict Prometheus parser; label cardinality bounded (verdict/quantile/window/measure/surface are closed sets — assert no unbounded `origin` explosion beyond the `Origin` enum).
- **Meta-session**: `runtime_meta_session` composes from `daemon_events`; all messages `material_origin=runtime_protocol`; **excluded from cost/authored-user counts**; queryable via read/MCP; assert it is NOT persisted to `index.db.sessions`.
- **`status --health`**: JSON model snapshot for each verdict; plain-text one-line banner; `--health` Click param declared **last** (positional-shift gotcha).
- **Dashboard**: HTML smoke — renders, **zero external URLs** (matches metrics.py zero-dep / Artifact-CSP posture), read-only, degrades on missing tables.
- Run via `devtools test <files>` (testmon inner loop); `/metrics` + status snapshots may need `render cli-output-schemas` / `render openapi` regen if the JSON envelope changes.

---

## 5. Bead breakdown (8, each with acceptance)

1. **ops.db `slo_samples` + `daemon_events` writer + `OPS_SCHEMA_VERSION→2` + sampler.** AC: table created on v2 rebuild; `record_slo_sample`/`record_daemon_event` in `ops_write.py`; sampler writes 5 measures per MEDIUM tick; retention GC bounded; rebuild plan documented; single coordinated bump.
2. **Measure-algebra reducer kernel** (pure module `daemon/slo_measures.py`: level/quantile/slope/eta/burn). AC: pure, unit-typed, `frozen_clock` tests, all edge cases incl. eta-undefined-on-growth.
3. **SLO evaluation + `[health.slo]` config + burn-rate alerts into MEDIUM tier.** AC: `SloThresholds.for_family` monotonic clamp; multi-window fast/slow alerts; immediate escalation/resolution + dedup (mirrors `cursor_lag_alert`); per-origin overrides.
4. **Construct-valid ingest-latency + convergence-lag measurement.** AC: live-tail origin scoping (bulk excluded, tested); freshness age = max over 3 derived-model lags; p50/p95/p99 correct on fixture; capture instant from acquired_at/mtime not message-ts.
5. **Runtime verdict + `polylogue status --health` + idle-vs-stalled.** AC: 6 verdict states; `HEALTHY_IDLE` when caught-up-quiet; `STALLED→ERROR`; structured JSON `RuntimeVerdict`; one-line banner; flag param last; daemon `/api/status?health=1` + local fallback.
6. **New Prometheus SLO series + /metrics contract test.** AC: 8 series with HELP+TYPE; strict-parse; bounded cardinality; degradation skeleton; wired after `_emit_embedding_metrics`.
7. **Synthetic runtime meta-session projector + read-surface wiring.** AC: `daemon_events`→synthetic session tree; `material_origin=runtime_protocol`; excluded from cost/authored; queryable via `read`/MCP `get_session_tree`; not written to index.db.
8. **Self-dashboard `GET /dashboard` from ops.db.** AC: zero external deps (inline CSS/JS/SVG); verdict banner + per-measure sparklines + latency bars + embedding ETA + burn gauges + meta-event feed; read-only via `open_readonly_connection`; degrades on missing tables.

Sequencing: 1→2 unblock everything; 3,4 parallel after 2; 5,6 after 3+4; 7,8 after 1 (meta-session/dashboard read `slo_samples`+`daemon_events`).

---

## 6. Top-3 risks

1. **Construct validity of "capture" instant.** There is no true user-send timestamp; the honest proxy is `acquired_at_ms`/source mtime (polylogue's acquire→durable), and message-timestamp would drag bulk-import history into the SLO. If mis-scoped, `ingest_latency` measures import backfill, not steady-state — producing both false alarms (old imports) and false calm (averaging away live spikes). *Mitigation:* hard live-tail origin allowlist + acquire-based capture + explicit bulk-exclusion, all pinned by construct-validity tests (Bead 4).
2. **Disposable-tier version bump wipes telemetry.** v2 forces "move aside and rebuild"; every future observability schema change resets `cursor_lag_samples` anomaly baselines *and* `slo_samples` slope/burn history, blinding the leading indicators for `baseline_window_days`. *Mitigation:* batch all Wave-5 ops additions into the single v2 bump; make slope/baseline reducers cold-start-tolerant (degrade to level-only until enough samples); document the re-scan cost. Accept sample loss as the tier's disposable contract.
3. **Sampler + reducer load on the sole SQLite writer.** Periodic sampling, slope/burn queries, and the dashboard/meta-session reads add work to the one writer + the health loop; `slo_samples` grows unbounded without GC. *Mitigation:* sample at existing MEDIUM cadence with bounded batch; retention GC reusing the `gc_cursor_lag_samples` pattern; every read path via `open_readonly_connection` so `/metrics`, `/dashboard`, and meta-session never block the writer; closed-set labels to cap `slo_samples`/Prometheus cardinality.

---

## [a6bbadcc01c5a8168] Wave-5 Spec — Multilingual (Polish) Search Recall

I have enough grounding. Here is the tight implementation spec.

---

# Wave-5 Spec — Multilingual (Polish) Search Recall

**Ground truth verified:**
- FTS DDL: `polylogue/storage/sqlite/archive_tiers/index.py:282-310` (`messages_fts`, `content=''`, `contentless_delete=1`, `tokenize='unicode61'`, default `remove_diacritics 1`). Duplicated in `polylogue/storage/fts/sql.py:10-21`.
- `search_text` is a VIRTUAL generated column (`index.py:201-207`); FTS is populated by triggers `messages_fts_ai/ad/au` and by explicit inserts in `fts/sql.py`.
- Query-side normalization: `normalize_fts5_query` / `escape_fts5_query` in `storage/search/query_support.py:54-106` — **no case/diacritic folding today**; relies entirely on unicode61.
- DSL: `archive/query/expression.py` — `QueryTextPredicate` (FTS leaf, line 1318), `QuerySemanticPredicate` (`near:text:`/`semantic:`, line 1329). FTS5 provider MATCH build: `storage/search_providers/fts5.py:78-96`.
- `blocks.language` (`index.py:194`) is the **programming-language** column (code-fence lang from parsers, e.g. `sources/parsers/drive_support_blocks.py`) — `prose_lang` MUST be a separate fact, not this.
- Index tier `INDEX_SCHEMA_VERSION = 24` (`index.py:36`), **rebuildable** regime.
- Vector lane: `embedding_model` default `voyage-4` (`config.py:246,992`) — already a multilingual model; hybrid fuse in `search_providers/hybrid.py`.

**Core technical fact driving the design:** `ł`/`Ł` (U+0142/U+0141) is a *precomposed* letter with an integral stroke, **not** a base+combining-diacritic sequence. It has **no canonical NFD decomposition**, so *neither* `remove_diacritics 1` *nor* `remove_diacritics 2` folds it. `remove_diacritics 2` fixes `ą ć ę ń ó ś ź ż` (combining-mark letters) but leaves `ł/Ł` unfolded. A Polish operator searching `zdlo` will never match `zdło`, and `łatwo` won't match a query typed `latwo`. This is the recall hole `remove_diacritics` alone cannot close.

---

## 1. Schema / DDL / tier / regime

**Tier:** all changes land in **`index.db`** (derived, rebuildable). No durable migration; per the durability-keyed regime, edit canonical DDL + rebuild plan, **never** an upgrade helper (`devtools lab policy schema-versioning` rejects one). Bump `INDEX_SCHEMA_VERSION 24 → 25`.

**1a. FTS tokenizer change (both DDL sites must stay identical):**
```
tokenize = 'unicode61 remove_diacritics 2'
```
Applied to `messages_fts` in `index.py:290` and the mirror in `fts/sql.py:19`. (These two strings are a known drift hazard — a lab check should assert they match; see risks.)

**1b. Stroke-fold at write time (the part `remove_diacritics` can't do).** Register a **deterministic** SQL function `pl_fold(text)` on the writer connection (`storage/sqlite/connection_profile.py` setup, before any trigger fires) and fold inside the FTS-insert path. Two equivalent placements — pick by trigger-vs-generated-column tradeoff:

- **Recommended:** keep `search_text` as-is; change the three triggers + `fts/sql.py` INSERTs to insert `pl_fold(new.search_text)` into the FTS `text` column. Contentless FTS stores only the folded token stream; original text is untouched in `blocks.text`. Reads need no function (contentless).
- Alternative (rejected): a second generated column `search_text_folded` — but generated columns require the function on *every* connection that reads the column, which is fragile for read-only openers. The trigger placement confines the dependency to the writer.

`pl_fold` maps only what unicode61 misses, keeping the two engines composable: `ł→l, Ł→l` (plus casefold to lowercase; unicode61 already lowercases, so `pl_fold` output is fed *before* unicode61 and just needs the stroke map + NFC). Vocabulary is a small closed dict, not a language model.

**1c. `prose_lang` facts (bead 0v9p), index tier, block grain:**
```sql
CREATE TABLE IF NOT EXISTS block_prose_lang (
    block_id     TEXT PRIMARY KEY REFERENCES blocks(block_id) ON DELETE CASCADE,
    session_id   TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    lang         TEXT NOT NULL,          -- BCP-47 primary subtag: 'pl','en','und'
    confidence   REAL NOT NULL,          -- 0..1
    is_mixed     INTEGER NOT NULL DEFAULT 0 CHECK (is_mixed IN (0,1)),
    detector     TEXT NOT NULL,          -- e.g. 'lingua'
    detector_version TEXT NOT NULL
) STRICT;
CREATE INDEX idx_block_prose_lang_session ON block_prose_lang(session_id, lang);
CREATE INDEX idx_block_prose_lang_lang ON block_prose_lang(lang) WHERE confidence >= 0.5;
```
Message/session rollups are **derived on read** (majority lang + `mixed` flag when >1 lang crosses a share threshold) — do **not** collapse a mixed message to one false language (0v9p AC). `lang='und'` for low-confidence/short blocks. Detector is pluggable (`lingua-py` recommended for short mixed text; `fast-langdetect` as a lighter option) and **not** part of any public contract. This table is populated by a new `ConvergenceStage` (see §3), so it's rebuildable like every other insight.

**User override / preference** stays in **`user.db`** as assertions (no user-tier schema change): a `metadata`/`correction`-kind assertion carrying preferred target language + a per-block/session language correction. Overrides never mutate `block_prose_lang` (source-derived) — they win at the read/projection boundary (0v9p AC: "override without altering source content").

---

## 2. Tokenization / folding algorithms (pseudocode)

The invariant the whole spec hangs on: **index-side and query-side must apply byte-identical folding before unicode61 sees the text.**

```
# Shared, deterministic, registered as SQL fn pl_fold() AND importable in Python.
STROKE_MAP = { 'ł':'l', 'Ł':'l', 'đ':'d','Đ':'d', 'ø':'o','Ø':'o' }  # extensible closed dict

def pl_fold(text):
    if text is None: return ''
    text = unicodedata.normalize('NFC', text)     # match hashing.py NFC discipline
    out = ''.join(STROKE_MAP.get(ch, ch) for ch in text)
    return out.casefold()   # unicode61 lowercases too; casefold makes query symmetry explicit
    # remove_diacritics 2 (in tokenizer) still handles ą/ć/ę/ń/ó/ś/ź/ż downstream.
```

**Index side** (trigger / fts insert):
```
INSERT INTO messages_fts(..., text) VALUES (..., pl_fold(new.search_text))
```

**Query side** (fold BEFORE escaping — insert one call in `normalize_fts5_query`, `query_support.py:54`):
```
def normalize_fts5_query(query):
    if blank(query): return None
    query = pl_fold(query)          # NEW — identical algorithm to index side
    fts = escape_fts5_query(query)  # unchanged; unicode61 rd=2 folds the rest at match time
    return None if fts == '""' else fts
```
`extract_match_terms` (`query_support.py:14`, feeds `matched_terms` highlight evidence) must fold identically or highlights desync from hits.

**DSL parity:** `QueryTextPredicate.text` (`expression.py:1318`) reaches FTS through the same `normalize_fts5_query`, so folding is inherited for free — but any *other* lowering path that builds a MATCH (grep for direct `MATCH` construction in `expression.py` pipeline lowerers, and `search_providers/fts5.py:81`) must route through the shared normalizer, not hand-roll escaping. That single choke point is the symmetry guarantee.

**Inflection (Polish is heavily inflected — `robić/robię/robił/robiony` share stem `robi`).** unicode61 has no stemmer and none is compiled (porter is out, and Polish needs a Hunspell-class stemmer regardless). Two bounded lanes rather than a stemmer:
- **Trigram fallback lane** — a second contentless FTS table `messages_fts_trigram USING fts5(text, tokenize='trigram', content='', contentless_delete=1)` fed by the same folded `pl_fold(search_text)`. Trigram gives substring/`LIKE`-grade recall so inflectional variants sharing a stem still overlap on trigrams; it also absorbs typos. Query router uses it as a **recall booster / fallback** when the primary lane yields few hits, unioned then re-ranked (bm25 primary, trigram as tiebreak). Cost: ~1.5–2× FTS index size on `search_text`; acceptable in a rebuildable tier.
- **Stopword handling (bounded, lower priority).** unicode61 has no stopword support. Polish stopwords (`i, w, z, na, że, się, to, ...`) mostly hurt precision, not recall, so treat as **query-side** stripping of a small closed Polish+English stoplist from *bare* multi-word queries only (never from quoted phrases or `near:` phrases). Do not drop them at index time — that would break exact-phrase matching. Ship the stoplist as data, gated behind detection that the query is Polish.

---

## 3. Rebuild plan

Pure derived-tier rebuild — no data migration, no backup manifest.

1. Land DDL: bump `INDEX_SCHEMA_VERSION → 25`; edit `messages_fts` tokenizer in **both** `index.py` and `fts/sql.py`; add `messages_fts_trigram` + its triggers; add `block_prose_lang`; register `pl_fold` in the connection profile so triggers resolve it.
2. Add rebuild-plan entry (canonical DDL + rebuild plan, per regime) and a new maintenance target (`maintenance/targets.py:40` style: `"messages_fts_trigram"`, `"block.prose_lang"`) so `polylogue ops reset --index && polylogued run` reconstructs everything.
3. `prose_lang` population is a new **`ConvergenceStage`** in `daemon/convergence_stages.py` with `check_many`/`execute_many` + `check_sessions`/`execute_sessions` variants; use the `false_means_pending` idiom to bound per-tick detection work and push backlog to `convergence_debt` (detection over a large archive is not free). Reuses the hot-file quiet-deferral already there.
4. Operator flow: `polylogue ops reset --index && polylogued run` (documented gotcha: schema mismatch triggers blue-green rebuild automatically). On a live 38 GB archive, batch this with any other pending index bumps from ready Beads (CLAUDE.md rule) — don't reset+reingest per isolated addition.
5. Post-rebuild verification: FTS freshness (`fts_freshness_state`, `index.py:312`) returns to `ready`; a golden Polish query returns expected hits.

---

## 4. Test strategy

Behavior/contract tests (not diff-memorializers), under `tests/unit/storage/` and `tests/unit/cli/`:

- **Folding symmetry law (highest value):** property test — for arbitrary block text, `search over pl_fold-derived index` and `query normalized via normalize_fts5_query` agree on a matching term. Assert `pl_fold` is **idempotent** and identical between the SQL fn and the Python import (round-trip a corpus through both, compare token sets).
- **Stroke recall:** seed a block `"zrobiłem to łatwo"`; assert queries `zrobilem`, `latwo`, `łatwo`, `zrobiłem` all hit. Baseline (current unicode61 rd=1) must *fail* the stroke cases — capture that as the reproduction the fix flips.
- **remove_diacritics 2 coverage:** `"gęś źdźbło"` matched by `ges`, `zdzblo`.
- **Mixed-language block (0v9p AC):** a block with a Polish sentence + an English code comment → `block_prose_lang.is_mixed=1`, not collapsed to one lang; message rollup reports mixed.
- **Low-confidence/`und`:** 2-word block → `lang='und'`, not a confident wrong guess.
- **User override (0v9p AC):** assertion-backed correction changes read-side language for a block without touching `block_prose_lang`; re-detection doesn't clobber the override.
- **Query predicate:** `find "x" prose_lang:pl` filters to Polish blocks; DSL `explain_query_expression` shows the predicate lowered.
- **Trigram fallback:** inflectional pair (`robić`/`robił`) — primary lane precision preserved, trigram lane recall documented (expected-recall assertion, not exact-set).
- **Vector cross-lingual (offload lane):** Polish query retrieves an English-content session via `near:text:` / hybrid — smoke-level, tolerant threshold.
- **No-porter guard** stays green; add a guard asserting the tokenizer string is `unicode61 remove_diacritics 2` in **both** DDL sites (drift lock).
- Clock hygiene via `frozen_clock`; run through `devtools test <files>` (testmon-affected), never blanket directories.

---

## 5. Bead breakdown (children under 0v9p / area:query+area:storage)

**B1 — `pl_fold` shared folding primitive + query-side symmetry.**
AC: deterministic `pl_fold` importable in Python and registered as SQL fn; `normalize_fts5_query` and `extract_match_terms` fold identically; idempotence + Python↔SQL parity property test green; no other MATCH-building path bypasses the normalizer (grep-verified). *Ship-first; unblocks everything.*

**B2 — Tokenizer `remove_diacritics 2` + stroke fold at index write.**
AC: both DDL sites carry `unicode61 remove_diacritics 2` and a drift-lock test; triggers + `fts/sql.py` insert `pl_fold(search_text)`; `INDEX_SCHEMA_VERSION → 25`; stroke + diacritic recall tests pass and the pre-fix baseline is shown failing; rebuild plan + maintenance target added; `render all --check` clean (grep `out of sync`).

**B3 — `block_prose_lang` facts table + detection convergence stage (bead 0v9p core).**
AC: block-grain facts with confidence/`is_mixed`/detector/version; message/session rollups derived on read without false collapse; `und` on low confidence; new `ConvergenceStage` with batch + session variants using `false_means_pending`; detector pluggable, not in public contract; tests for mixed/low-confidence.

**B4 — `prose_lang:` query predicate + projection default.**
AC: DSL predicate filters by source language across CLI/API/MCP; lowering shown in `explain_query_expression`; projection can pick candidate translation target from facts (feeds 4smp/rlsb); generated CLI reference + output schemas regenerated.

**B5 — User language override/preference via user.db assertions.**
AC: assertion-backed correction/preference overrides derived detection at read boundary without mutating `block_prose_lang` or source; re-detection idempotent against override; test coverage for override + no-translation-created-by-detection (0v9p AC).

**B6 — Trigram fallback recall lane.**
AC: `messages_fts_trigram` fed by folded text; query router unions + re-ranks (bm25 primary) when primary recall is low; inflection pair recall documented; index-size cost noted; rebuild/maintenance wired.

**B7 (optional) — Polish stopword query-side handling.**
AC: closed Polish+English stoplist stripped from *bare multi-word* queries only, gated on Polish detection; quoted/`near:` phrases untouched; precision test.

**B8 (optional) — Cross-lingual vector offload routing.**
AC: when query prose_lang differs from archive-dominant lang or FTS recall is low, boost/route to the `voyage-4` semantic lane in the hybrid fuse; cross-lingual smoke test (PL query → EN hit).

Suggested landing order: **B1 → B2 → B3 → B4 → B5**, then B6, then B7/B8. B1+B2 are one coherent PR (folding is meaningless split from the tokenizer change); B3+B4 another; B5 small; B6 standalone.

---

## 6. Top-3 risks

1. **Custom-tokenizer temptation vs. pure-Python constraint.** The "correct" FTS answer is a registered fts5 tokenizer, but that needs the `fts5_api` C pointer (APSW-class), which violates Polylogue's stdlib-`sqlite3` / no-native-deps posture. The `pl_fold`-at-write-time design is the deliberate workaround; its Achilles' heel is that the folding fn must be registered on the **writer connection before any trigger fires** — if a writer opens without it, INSERTs raise `no such function: pl_fold` and ingest stalls. Mitigation: register in `connection_profile` unconditionally for the index tier and add a startup assertion.

2. **Query/index folding drift = silent recall loss.** The whole design is a *symmetry contract* across two code paths (SQL `pl_fold` in triggers, Python `pl_fold` in `normalize_fts5_query`) and two DDL strings (`index.py` + `fts/sql.py`). Any divergence — a stroke added to one map, `remove_diacritics 2` in one DDL only, a MATCH-builder that skips the normalizer — degrades recall invisibly (no error, just missing hits). Mitigation: single shared `pl_fold` source, the Python↔SQL parity property test (B1), and the DDL drift-lock test (B2).

3. **Live 38 GB rebuild + detection cost.** Changing the tokenizer forces a full `messages_fts` rebuild; adding the trigram lane ~doubles that FTS's size; `prose_lang` detection runs a language model over every block. On the live archive this is a heavy convergence burst. Mitigation: batch this index bump with other pending ones (don't reset per addition), lean on `false_means_pending` + hot-file quiet deferral so detection backlogs into `convergence_debt` instead of blocking ingest, and verify `fts_freshness_state` returns `ready` before declaring done. **Not verified live** — the rebuild-time and index-size figures are estimates; measure on first real reset.

*(Not run: no code/beads were modified per instructions; all findings are from source inspection of the cited files.)*

---

## [a8cdc6b65eb313dbd] Spec: `spec-cards` + `trajectory-quality-index`

I have enough grounding. Here is the tightly-scoped implementation spec.

---

# Spec: `spec-cards` + `trajectory-quality-index`

## Grounding (real files)

| Concern | Location | What it gives us |
|---|---|---|
| Commit attribution | `polylogue/insights/session_commit.py`, `session_commits` table `index.py:502` | `(session_id, commit_sha, repo_id, detection_type∈{time_window,file_overlap,explicit_ref,origin_reported}, confidence, evidence_json)` + `repos`/`session_repos` (branch, root_path) |
| Terminal state | `archive/session/runtime.py:220` `_terminal_state` | `clean_finish / error_left / tool_left / question_left / unknown` + confidence + evidence — the **outcome_signal** and **abandonment** primitive |
| Latency | `archive/semantic/timing.py:159` `SessionLatencyProfileFacts` | `median/p90/max_tool_call_ms`, `stuck_tool_count`, `median_agent/user_response_ms`, `timing_provenance` |
| Pathology | `insights/pathology.py` (`PATHOLOGY_DETECTOR_VERSION=4`) | `wasted_loop`, `stale_context` findings w/ `EvidenceRef` |
| Phases | `archive/phase/extraction.py` | **time-gap segments, `kind` deliberately removed** (5-min idle gaps). `duration_ms`, `tool_counts`, `word_count` |
| Tool-error rate | `blocks.tool_result_is_error` via `actions` view (`insights/transforms.py`) | structural provider-reported failure |
| Corrections | `user.db` `assertions` `kind=correction` | operator-recorded correction density |
| Insight registration | `insights/registry.py` `INSIGHT_REGISTRY`, `register(InsightType(...))` | descriptor pattern |
| Measure registry | **does not exist yet** — `bd:9l5.7` (`analytics/stats.py` + `MeasureSpec`, tiers structural/provider-reported/derived/heuristic) | TQI must register here → **blocked-by 9l5.7 slice-1** |
| Eval export | `bd:fs1.5` (Atropos/eval JSONL downstream of canonical archive) | spec-card export target |
| Tier | index.db is **derived/rebuildable, ver 24** | no migration chain; DDL edit + rebuild |

---

## (1) Schema / DDL + tier

Both surfaces are **derived → `index.db` (rebuildable), bump `user_version` 24→25**. No numbered migration; edit canonical DDL in `storage/sqlite/archive_tiers/index.py` + rebuild plan (`polylogue ops reset --index && polylogued run`).

```sql
-- spec_cards: one portable benchmark projection per session (STRICT)
CREATE TABLE IF NOT EXISTS spec_cards (
    session_id            TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    repo_id               TEXT REFERENCES repos(repo_id) ON DELETE SET NULL,
    initial_repo_sha      TEXT,          -- first-parent of earliest attributed commit / provider-reported HEAD; NULL if unknown
    final_repo_sha        TEXT,          -- latest attributed commit
    intent_text           TEXT,          -- first human_authored message (single field, NOT transcript)
    intent_redacted       INTEGER NOT NULL DEFAULT 0 CHECK(intent_redacted IN (0,1)),
    acceptance_signal     TEXT,          -- 'pr_merged' | 'tests_pass' | 'explicit_ref' | 'none'
    acceptance_ref_json   TEXT NOT NULL DEFAULT '{}',   -- ObjectRef(s): github-pr / commit
    tools_used_json       TEXT NOT NULL DEFAULT '{}',   -- {category: count} from tool_call_count_by_category
    outcome_signal        TEXT,          -- mirrors terminal_state
    final_diff_ref        TEXT,          -- ObjectRef(kind=diff): initial_sha..final_sha; NULL if repo absent
    diff_stat_json        TEXT NOT NULL DEFAULT '{}',   -- {files, insertions, deletions} (portable w/o diff body)
    evidence_tier         TEXT NOT NULL CHECK(evidence_tier IN ('structural','provider-reported','derived','heuristic')),
    completeness          REAL NOT NULL CHECK(completeness BETWEEN 0 AND 1),  -- fraction of card fields grounded
    evidence_json         TEXT NOT NULL DEFAULT '{}',   -- EvidenceRefs per field
    card_version          INTEGER NOT NULL DEFAULT 1,
    materializer_version  INTEGER NOT NULL,
    materialized_at       TEXT NOT NULL DEFAULT '',
    source_updated_at     TEXT,
    search_text           TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_spec_cards_tier ON spec_cards(evidence_tier);
CREATE INDEX IF NOT EXISTS idx_spec_cards_repo ON spec_cards(repo_id) WHERE repo_id IS NOT NULL;

-- session_quality_index: composed TQI + transparent components (STRICT)
CREATE TABLE IF NOT EXISTS session_quality_index (
    session_id            TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    tqi_score             REAL CHECK(tqi_score IS NULL OR tqi_score BETWEEN 0 AND 1),
    responsiveness_sub    REAL,   -- from latency (stall / p90)
    correction_sub        REAL,   -- correction density
    completion_sub        REAL,   -- terminal_state → abandonment
    tool_reliability_sub  REAL,   -- 1 - tool_result_is_error rate
    fragmentation_sub     REAL,   -- reoperationalized "phase thrash" (heuristic)
    component_tiers_json  TEXT NOT NULL DEFAULT '{}',  -- per-subscore evidence_tier
    weights_json          TEXT NOT NULL DEFAULT '{}',  -- documented weight vector (measure_version-pinned)
    coverage_json         TEXT NOT NULL DEFAULT '{}',  -- n per component + timing_provenance
    insufficient_coverage INTEGER NOT NULL DEFAULT 0 CHECK(insufficient_coverage IN (0,1)),
    measure_version       INTEGER NOT NULL,            -- pins formula; bump invalidates cross-session comparability
    materializer_version  INTEGER NOT NULL,
    materialized_at       TEXT NOT NULL DEFAULT '',
    evidence_json         TEXT NOT NULL DEFAULT '{}',
    search_text           TEXT NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX IF NOT EXISTS idx_sqi_score ON session_quality_index(tqi_score DESC) WHERE tqi_score IS NOT NULL;
```

`tqi_score` is nullable + `insufficient_coverage` flag so the composition layer can **refuse to emit a bare number** when required coverage is missing (9l5.7 discipline).

---

## (2) Derivation algorithms (pseudocode)

Both materialize in the DaemonConverger insight stage from already-computed inputs (session_profile, latency_profile, pathology, session_commits, corrections) — **no re-parse, no I/O except git for diff**.

### spec-card

```
build_spec_card(session, profile, latency, commits, corrections):
    repo    = dominant session_repos row (highest observed / branch)
    landed  = [c for c in commits sorted by created_at
               if c.detection_type in ('explicit_ref','origin_reported')  # trust only high-confidence
               or c.confidence >= 0.5]
    final_sha    = landed[-1].commit_sha if landed else None
    initial_sha  = git first-parent(landed[0]) if repo present
                   else provider_reported_head(session)  # Codex/CC JSONL git ref, if any
                   else None
    intent  = first message where material_origin == human_authored   # SINGLE field
    accept, accept_ref = classify_acceptance(commits, github_pr_refs(session))
        # pr_merged (structural, needs live gh/lynchpin) > explicit_ref > tests_pass(from observed test_passed events) > none
    tools   = profile.tool_call_count_by_category
    outcome = profile.terminal_state                      # clean_finish / error_left / ...
    if repo present and initial_sha and final_sha:
        diff_stat = git diff --stat initial_sha..final_sha  (scoped to session_repos paths)
        diff_ref  = ObjectRef(kind='diff', object_id=f"{initial_sha}..{final_sha}")   # reference, body NOT stored
    tier = worst-tier over grounded fields
           (structural if sha+diff+pr_merged from git/gh; heuristic if intent-only)
    completeness = grounded_field_count / 6
    emit row (+ EvidenceRef per field in evidence_json)
```

**No-transcript-leakage:** the card stores exactly six derived fields; `intent_text` is one turn, gated by `--redact-intent` (stores hash + length only, `intent_redacted=1`). Export path (fs1.5) never joins `messages`/`blocks`. A leakage-guard test asserts the export JSONL schema contains no `search_text`/block bodies.

### trajectory-quality-index

Each subscore ∈ [0,1], higher = healthier. All monotone, documented.

```
responsiveness_sub = clamp01(1 - stuck_tool_count / max(1, total_tool_calls))
                     * decay(latency.p90_tool_call_ms, ref=300_000ms)     # provider-reported tier
correction_sub     = clamp01(1 - corrections_recorded / max(1, authored_user_turns))  # derived; SPARSE confound
completion_sub     = { clean_finish:1.0, question_left:0.6, tool_left:0.3,
                       error_left:0.2, unknown:NULL }[terminal_state]
                     weighted by terminal_state_confidence                # derived
tool_reliability_sub = wilson_lower(  # 9l5.7 primitive, not raw rate
                          successes = tool_results - error_results,
                          n = tool_results )                              # structural
fragmentation_sub  = clamp01(1 - (phase_count - 1) / max(1, active_minutes/5))  # HEURISTIC — see caveat
                     # reoperationalized: phases are time-gap segments, NOT intent switches

present = subscores that are non-NULL and meet component coverage
if required_coverage_unmet(present):       # e.g. no timing_provenance, terminal unknown
    tqi_score = NULL; insufficient_coverage = 1
else:
    w = documented weight vector (pinned by measure_version)
    tqi_score = weighted_geometric_mean(present, w)   # geometric: one broken component drags score, no averaging-away
component_tiers = {sub: its evidence_tier}
```

Registered as a **`MeasureSpec`** (9l5.7): `construct="collaboration health (observational)"`, `evidence_tier=derived` (dominated by heuristic `fragmentation_sub`), `required_coverage=["timing_provenance","terminal_state_confidence>=0.5"]`, `confounds=["task difficulty","correction under-recording","abandonment≠failure","Goodhart under RL"]`. `insight_rigor_audit` picks it up automatically.

---

## (3) Migration

Derived tier ⇒ **no numbered SQL migration**. Steps:
1. Edit canonical DDL (`archive_tiers/index.py`), bump `INDEX_SCHEMA_VERSION` 24→25.
2. Add rebuild-plan entry; `devtools lab policy schema-versioning` must pass (rejects any upgrade-helper for derived tiers).
3. New module(s) under `polylogue/` ⇒ regenerate topology: `devtools render topology-projection && devtools render topology-status`, commit `docs/plans/topology-target.yaml` + `docs/topology-status.md`.
4. New `InsightType`/MCP tool ⇒ update `EXPECTED_TOOL_NAMES` + tool contract; regenerate `render openapi` + `render cli-output-schemas` if any enum/AssertionKind touched.
5. Deploy: `polylogue ops reset --index && polylogued run` rebuilds spec_cards + session_quality_index from source (durable tiers untouched).

---

## (4) Test strategy

- **Property (`tests/property`, hypothesis):** `tqi ∈ [0,1]`; monotonicity (more stalls ⇒ lower responsiveness_sub; more corrections ⇒ lower correction_sub); NULL-propagation when a component is unavailable; `insufficient_coverage` forces NULL score.
- **Wilson coverage** property for `tool_reliability_sub` (interval coverage on synthetic distributions — shared with 9l5.7).
- **Golden spec-cards** on `corpus_seeded_db`: seed a session with an explicit-ref commit + PR ref ⇒ assert `evidence_tier='structural'`, `acceptance_signal='pr_merged'`, all six fields grounded; seed an intent-only session ⇒ `tier='heuristic'`, `completeness` low.
- **Leakage-guard (security-adjacent, protected):** assert exported spec-card JSONL contains none of {block bodies, `search_text`, non-intent message text}; `--redact-intent` zeroes `intent_text`.
- **Determinism:** same session ⇒ byte-identical card (content-hash idempotency; re-materialize is a no-op).
- **Rebuild parity:** `ops reset --index` reproduces identical spec_cards/SQI rows.
- **Fragmentation caveat regression:** a single-burst session and a fragmented session with equal work assert the intended fragmentation_sub ordering (documents the heuristic).
- Use `frozen_clock`; run via `devtools test <files>` (testmon), not blanket directories.

---

## (5) Bead breakdown (7, with acceptance)

Parent/blocking: **spec-card cluster → `fs1.5`**; **TQI cluster → blocked-by `9l5.7` slice-1** (MeasureSpec shape).

1. **`spec-cards.1` — spec_cards table + derivation (index tier).** AC: DDL bumped 24→25; `build_spec_card` materializes 6 fields from profile+commits; determinism + rebuild-parity tests green; topology regenerated. *(S/M)*
2. **`spec-cards.2` — acceptance/outcome grounding + evidence tiering.** AC: `pr_merged` via gh/lynchpin ⇒ structural; `explicit_ref` ⇒ provider-reported; intent-only ⇒ heuristic; `completeness` computed; per-field EvidenceRefs. *(M)*
3. **`spec-cards.3` — portable export (JSONL, fs1.5-aligned) + leakage guard.** AC: `polylogue analyze spec-cards --export` emits benchmark JSONL joining no message bodies; leakage-guard + `--redact-intent` tests green. *(M)* — **Ref fs1.5**
4. **`tqi.1` — MeasureSpec registration for TQI (contract-first).** AC: TQI declared as a `MeasureSpec` with construct, formula ref, `evidence_tier`, `required_coverage`, confounds; `insight_rigor_audit` audits it. *(S)* — **blocked-by 9l5.7 slice-1**
5. **`tqi.2` — session_quality_index table + composition.** AC: five subscores + geometric composition; NULL/`insufficient_coverage` gating; refuses bare render without coverage; property tests (range/monotonicity/NULL-prop). *(M)*
6. **`tqi.3` — fragmentation reoperationalization + caveat surface.** AC: phase-thrash implemented as temporal-fragmentation (heuristic tier), lowest weight, confound documented in MeasureSpec; ordering regression test. *(S)*
7. **`tqi.4` — dual-use surfaces: dashboard read + RL reward-shaping adapter.** AC: MCP/CLI `analyze trajectory-quality` renders score + component tiers + CI (Wilson/bootstrap from 9l5.7); a `--reward-shaping` export emits per-session scalar with documented sign/normalization + Goodhart caveat header. EXPECTED_TOOL_NAMES + tool contract updated. *(M)* — **Ref h10/stc**

---

## (6) Top-3 risks

1. **Spec-card reproducibility depends on weak commit attribution + local repo presence.** `session_commits` is heuristic (file-overlap ≥0.3, ±2h window); `initial_repo_sha`/`final_diff` need the repo checked out at the right SHA. Most sessions won't yield a reproducible benchmark item, and a wrong SHA silently produces a *plausible but false* benchmark. **Mitigation:** emit `structural`-tier cards only from `explicit_ref`/`origin_reported`+`pr_merged` evidence; gate the fs1.5 export on tier; `completeness` makes partial cards honestly partial rather than fabricated.

2. **"Phase thrash" is not measurable from the phase model.** `SessionPhase.kind` was deliberately removed — phases are 5-min idle-gap segments, not intent classes. Composing an intent-switching signal from them imports an invalid construct that can dominate TQI. **Mitigation:** reoperationalize as *temporal fragmentation*, tag `heuristic`, give it the lowest weight, document the confound in the MeasureSpec, and keep it separable/removable in v1.

3. **TQI-as-RL-reward invites Goodhart / reward-hacking.** It is an observational composite confounded by task difficulty; optimizing it directly rewards `clean_finish` theater and suppressed corrections (correction_sub *rises* when the operator stops bothering to correct). **Mitigation:** ship strictly as a reward-*shaping* term (never sole reward), keep component transparency in every payload, pin `measure_version` so a formula change invalidates comparability, and let the 9l5.7 registry refuse bare single-number rendering.

**Cross-cutting caveat to carry in the MeasureSpec:** TQI mixes structural (`tool_reliability`), provider-reported (`responsiveness`), derived (`correction`,`completion`), and heuristic (`fragmentation`) tiers into one scalar — the composite's tier is the worst component, and the per-component `component_tiers_json` must always ride alongside the score.

---

## [a9f822915a0320ef6] Spec: Read-Access-Log + Attention Substrate

I have full grounding. Delivering the spec.

---

# Spec: Read-Access-Log + Attention Substrate

Grounded in: `ops.py` (OPS DDL), `ops_write.py` (write helpers), `daemon/events.py` (the multi-writer ops.db pattern — the exact template), `bootstrap.py` (disposable-tier init + self-healing shims), `api/archive.py:2713` + `mcp/server_tools.py:274/347/854` (read entrypoints), bead **polylogue-37t.11** (the context-scheduler `ContextSource`/ledger spec — no doc file exists; it's the convergent design), and user.db `AssertionKind.SAVED_QUERY`/`RECALL_PACK` (dead-memory targets).

**Key architectural fact that shapes everything:** `daemon/events.py:emit_daemon_event` proves that ops.db is a *multi-writer disposable tier* — it opens ops.db directly via `open_daemon_connection` (WAL + busy_timeout), self-heals its own `CREATE TABLE IF NOT EXISTS`, inserts, commits, closes — **without routing through the daemon's sole-writer path** (that constraint is index.db/source.db only). The read-access-log writes exactly the same way: any surface (CLI/MCP/API) writes its own access rows inline. This is the load-bearing decision.

---

## 1. Schema / DDL + tier

**Tier: `ops.db` (disposable).** New table appended to `OPS_DDL` in `polylogue/storage/sqlite/archive_tiers/ops.py`. A **debounced aggregate** (not an append-only firehose) — one row per `(item_ref, reader_session, surface, access_kind)`, carrying a **decayed counter** so the attention signal is a single float, and growth is bounded to O(items × readers × surfaces).

```sql
CREATE TABLE IF NOT EXISTS context_read_access (
    item_ref            TEXT NOT NULL,          -- normalized public ref (session:/message:/block:/assertion:/recall-pack:/saved-view:)
    item_kind           TEXT NOT NULL,          -- denormalized ObjectRef.kind for cheap GROUP BY
    reader_session      TEXT,                   -- best-effort acting agent/session id; NULL when unknown
    surface             TEXT NOT NULL CHECK(surface IN ('cli','mcp','api','daemon','web')),
    access_kind         TEXT NOT NULL CHECK(access_kind IN
                          ('resolve_ref','read','get_session','get_messages','search_hit','context_inject')),
    first_accessed_at_ms INTEGER NOT NULL,
    last_accessed_at_ms  INTEGER NOT NULL,
    access_count         INTEGER NOT NULL DEFAULT 1 CHECK(access_count >= 0),
    decayed_score        REAL NOT NULL DEFAULT 1.0,   -- exponential-decay counter (half-life λ)
    score_updated_at_ms  INTEGER NOT NULL,            -- when decayed_score was last renormalized
    PRIMARY KEY (item_ref, reader_session, surface, access_kind)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_cra_item        ON context_read_access(item_ref, last_accessed_at_ms DESC);
CREATE INDEX IF NOT EXISTS idx_cra_kind_recent ON context_read_access(item_kind, last_accessed_at_ms DESC);
CREATE INDEX IF NOT EXISTS idx_cra_reader      ON context_read_access(reader_session, last_accessed_at_ms DESC);
```

Notes:
- `item_ref` is the **normalized** form (`resolve_ref` already produces `normalized_ref = parsed.format()`); `item_kind` = `object_ref.kind`. **Refs only — never content** (SYSTEM trust class per 37t.11).
- `reader_session` is nullable: grep confirms **no current agent-session identity plumbed into MCP hooks**. Populate best-effort from `POLYLOGUE_READER_SESSION` env / MCP hooks context once the coordination program (**polylogue-s7ae**) lands agent identity; do not block on it.
- `context_inject` is captured but **deliberately excluded from the attention signal** (see risk 3 — prevents the scheduler reinforcing its own injections).

---

## 2. Write-path + attention algorithms (pseudocode)

### 2a. Writer — models `emit_daemon_event` exactly (self-healing, best-effort, fire-and-forget)

```
# ops_write.py  (new helpers, sibling to record_daemon_stage_event)
_READ_ACCESS_DDL = "<the CREATE TABLE/INDEX above>"
_PROCESS_DEBOUNCE: dict[key, last_ms] = {}      # in-process, per (item_ref,reader,surface,kind)
DEBOUNCE_MS = 60_000
HALF_LIFE_MS = 14 * 86_400_000                  # λ = ln2 / HALF_LIFE_MS

def record_context_read_access(item_ref, item_kind, surface, access_kind,
                               reader_session=None, now_ms=clock.now_ms()):
    key = (item_ref, reader_session, surface, access_kind)
    # (1) IN-PROCESS debounce — kills read-amplification storms before touching sqlite
    if now_ms - _PROCESS_DEBOUNCE.get(key, 0) < DEBOUNCE_MS:
        return
    _PROCESS_DEBOUNCE[key] = now_ms
    try:                                        # best-effort: a read must NEVER fail on log write
        conn = _ensure_read_access_db()         # initialize_archive_database(ops) + executescript(_READ_ACCESS_DDL)
        # (2) decayed-counter UPSERT: renormalize old score to now, then +1
        conn.execute("""
          INSERT INTO context_read_access
            (item_ref,item_kind,reader_session,surface,access_kind,
             first_accessed_at_ms,last_accessed_at_ms,access_count,decayed_score,score_updated_at_ms)
          VALUES (?,?,?,?,?,?,?,1,1.0,?)
          ON CONFLICT(item_ref,reader_session,surface,access_kind) DO UPDATE SET
            last_accessed_at_ms = excluded.last_accessed_at_ms,
            access_count        = access_count + 1,
            decayed_score       = decayed_score * exp(-LN2/HALF_LIFE_MS *
                                    (excluded.last_accessed_at_ms - score_updated_at_ms)) + 1.0,
            score_updated_at_ms = excluded.last_accessed_at_ms
        """, (item_ref,item_kind,reader_session,surface,access_kind,now_ms,now_ms,now_ms))
        conn.commit()
    except sqlite3.Error:
        log.debug("read-access log write skipped")   # swallow: readonly FS / ops reset mid-flight
    finally:
        conn.close()
```
> `exp` is registered as a SQLite scalar function on the connection (one-liner in `connection_profile`), or the decay is computed in Python and passed as a bound param — either works; Python-side is simpler and avoids a per-connection function registration.

### 2b. Instrumentation — only at **surface entrypoints**, never internal helpers

One call site per public read path, guarded by the in-process debounce (so `read --view transcript`, which internally recomposes lineage and resolves hundreds of parent refs via `1e4a69438`, still logs **one** access for the session the user asked for):

```
# api/archive.py::resolve_ref  (after producing normalized_ref + object_ref)
record_context_read_access(normalized_ref, object_ref.kind, surface="api", access_kind="resolve_ref", reader_session=...)

# mcp/server_tools.py::resolve_ref / get_session / get_messages  → surface="mcp"
# cli read verb                                                  → surface="cli"
```
Do **not** instrument the internal `_resolve_session_object_ref`/lineage-composition helpers — only the top-level verb the surface invoked.

### 2c. Attention read model — decay-at-read, normalized to [0,1]

```
def attention_score(item_ref, now_ms) -> float:      # feeds the scheduler
    rows = SELECT decayed_score, score_updated_at_ms
           FROM context_read_access
           WHERE item_ref = ? AND access_kind != 'context_inject'   # exclude self-injections
    raw = Σ_rows decayed_score * exp(-LN2/HALF_LIFE_MS * (now_ms - score_updated_at_ms))
    return raw / (raw + 1.0)          # squash to (0,1); 0 == never read → dead memory candidate
```

### 2d. Context-scheduler wiring (37t.11 `ContextSource`)

Attention is a **SYSTEM-class ranking bonus** — it can only reorder candidates, never inject prose:
```
# inside a ContextSource.propose() (recall leg mhx.4, lessons leg rvh)
item.relevance = base_similarity * (1 + W_ATTN * attention_score(item.ref, now))
# W_ATTN small (e.g. 0.25). Proven-reread refs float up; the scheduler still owns budget.
```

### 2e. Dead-memory detection

```
# never-opened recall packs / saved views (pure access-log join)
saved = list_assertions_by_kind(user, {RECALL_PACK, SAVED_QUERY})   # user.db
for a in saved:
    ever_read = EXISTS(SELECT 1 FROM context_read_access
                       WHERE item_ref = a.target_ref
                         AND access_kind IN ('resolve_ref','read'))
    if not ever_read and age(a) > STALE_AFTER: flag "dead-memory: never opened"

# zero-result saved views → needs execution (companion probe, own bead):
#   convergence stage runs each SAVED_QUERY expression, records result_count;
#   result_count == 0 over K probes → flag "dead-memory: query returns nothing"
```

---

## 3. Migration

Classification: **additive-derived, disposable tier** — no numbered migration, no backup manifest, no forced rebuild.

1. Append `context_read_access` DDL to `OPS_DDL` (covers fresh archives via `initialize_archive_tier`).
2. Add `_ensure_ops_read_access_table(conn)` to the `tier is ArchiveTier.OPS` branch in `bootstrap.py` (idempotent `CREATE TABLE/INDEX IF NOT EXISTS`), mirroring `_ensure_ops_cursor_lag_sample_columns`.
3. The writer's `_ensure_read_access_db()` **self-heals inline** (executescript on every open, exactly like `daemon/events.py`), so a live ops.db at version 1 gains the table on first write with **no `OPS_SCHEMA_VERSION` bump** — `daemon_events` itself never bumped for this reason.
4. Optional: bump `OPS_SCHEMA_VERSION 1→2` for canonical cleanliness; because ops is disposable the mismatch path is `polylogue ops reset --... && polylogued run`, but the self-healing writer makes even that unnecessary. **Recommend: no version bump.**
5. `devtools lab policy schema-versioning` must stay green (disposable tier, no upgrade helper introduced).

---

## 4. Test strategy (behavior/invariant, `frozen_clock` — not DDL memorialization)

- **Debounce coalescing:** N rapid accesses to same key within `DEBOUNCE_MS` → `access_count` reflects one increment (in-process) / bounded rows; distinct keys stay separate.
- **Read-amplification guard:** `read --view transcript` on a deep lineage session emits exactly one `context_read_access` row for the requested session (not one per recomposed parent ref).
- **Decay math (property):** `attention_score` monotonically decreasing in elapsed time; at exactly one half-life a single-access score halves (±ε); never negative; unread item → 0.
- **Instrumentation coverage:** `resolve_ref`/`get_session`/`get_messages` each write one row with correct `surface`/`access_kind`; failures (unresolved ref) still log or deliberately don't — assert the chosen contract.
- **Best-effort isolation:** a write against a readonly/removed ops.db raises internally but the read result is unaffected (inject `sqlite3.OperationalError`).
- **Dead-memory:** seed a recall pack, never access → flagged; access once → not flagged; zero-result saved-view probe flags a query with no hits.
- **Concurrency:** two processes writing overlapping keys under WAL → no corruption, counts reconcile (busy_timeout honored).
- **Disposability:** `ops reset` drops the table; next daemon/read run recreates it via self-heal.
- **Trust class (feeds 37t.11 red-team):** attention never introduces content into an assembled preamble — only reorders; assert assembled output byte-set unchanged except ordering.

Run via `devtools test <files>` (testmon-affected), not blanket directories.

---

## 5. Bead breakdown (dependencies noted)

1. **ops schema + writer** (size:S). `context_read_access` in `OPS_DDL` + `_ensure_ops_read_access_table` + `record_context_read_access` (in-process debounce, decayed-counter upsert, best-effort). **AC:** table self-heals on existing ops.db with no version bump; debounce collapses repeat writes; decay upsert renormalizes correctly; write failure never propagates.
2. **Instrument read entrypoints** (size:M, dep 1). Call sites in `api/archive.py` (`resolve_ref`, `get_session`, `get_messages_paginated`), `mcp/server_tools.py` (`resolve_ref`,`get_session`,`get_messages`), CLI `read`. **AC:** each public read logs exactly one row with correct surface/kind; transcript recompose logs one, not N; no internal-helper double-logging.
3. **Attention read model** (size:S, dep 1). `attention_score(item_ref)` + `list_attention(kind,limit)` in `ops_write.py`, decay-at-read, `context_inject` excluded. **AC:** score in (0,1), 0 for unread, half-life property holds; injection-kind rows excluded.
4. **Scheduler wiring** (size:M, dep 3, **blocked by polylogue-37t.11**). Attention as SYSTEM-class ranking bonus in recall/lessons `ContextSource.propose()`. **AC:** proven-reread ref ranks above equal-similarity unread ref; never alters injected content, only order; ledger records unchanged trust class.
5. **Dead-memory: never-opened** (size:S, dep 1). Access-log ⋈ user.db `RECALL_PACK`/`SAVED_QUERY`; surface via CLI `analyze` + MCP tool. **AC:** never-opened pack past staleness flagged; opening clears the flag; excludes the creating write.
6. **Saved-view liveness probe** (size:M, dep 5). Convergence stage executes each saved view, records `result_count`; zero over K probes → flagged. **AC:** a saved view whose query returns nothing is flagged after K probes; a live view is not; runs under `false_means_pending` deferral discipline.
7. **"Archive tracks its own use" analytics** (size:M, dep 1). Most-re-read sessions, per-surface/per-kind rollups; CLI + one MCP tool (+`EXPECTED_TOOL_NAMES`/contract update). **AC:** rollup matches seeded access counts; MCP discovery test passes; refs-only output.
8. *(optional, size:S, dep polylogue-s7ae)* **reader_session identity** from MCP hooks/coordination + `POLYLOGUE_READER_SESSION` env fallback. **AC:** when identity present, per-reader attention partitions; absent → NULL, aggregate still works.

---

## 6. Top-3 risks

1. **Read-amplification self-write storm.** Lineage/transcript composition (`1e4a69438`, iterative recompose) resolves hundreds of parent refs per single user read; instrumenting `resolve_ref` naively would write a burst per user action and could recursively log the composition's own internal resolutions. **Mitigation:** in-process debounce + instrument **only surface entrypoints** (not internal `_resolve_*` helpers) + `access_kind` separation so internal resolves are never the logged unit.
2. **ops.db write pressure & read-path fragility.** Adding a hot write to every read increases WAL churn on the disposable tier and risks a read failing when ops.db is readonly / mid-`ops reset`. **Mitigation:** fire-and-forget best-effort writes (swallow every `sqlite3.Error`), short busy_timeout, bounded aggregate (not append-only), decayed counter keeps row count O(items×readers×surfaces). ops.db is *already* multi-writer (daemon events, cursor, health) — no new concurrency regime.
3. **Attention feedback loop + behavioral-surveillance surface.** Boosting recently-read items into context that then get re-read is runaway reinforcement; and a per-reader access log is a sensitive behavioral trace. **Mitigation:** exclude `context_inject` from the signal (scheduler can't reinforce itself), decay half-life + small `W_ATTN` cap, SYSTEM trust class (ranking-only, no prose injection — satisfies 37t.11's injection-hole constraint), and keep everything in the **disposable** tier (resettable, no durability, refs-only, `reader_session` optional).

---

## [a087b4bbe668eece7] Polylogue — The Honest Critical Path (Wave-5 synthesis)

I have enough grounding. Here is the synthesis.

---

# Polylogue — The Honest Critical Path (Wave-5 synthesis)

Grounded in `bd list` (50 open, 9 epics), the master brainstorm doc (170 ideas / 66 prompts), the swarm2 design set, and the deep-audit findings. I did not read the huge task outputs; I read the master doc, the audit, `bd ready --json`, and spot-checked the daemon/cost code the red-team implicates.

## 1. The minimal high-value core — and where the red-team is right vs wrong

**The red-team's 80/20 claim is directionally correct and should reshape sequencing.** The realized, load-bearing value of Polylogue today is exactly two things, both confirmed in project memory: **(a) agent-memory continuity** (assertions injected into context, the MCP recall surface — "assertions > CLAUDE.md") and **(b) honest cost/usage visibility** (material_origin-gated user-word accounting, subscription-vs-API dual view). Everything the operator actually *cites* traces to one of those two.

**Where the red-team is wrong: its proposed core is too small and skips the moat.** "SessionEnd-hook copy + FTS + 3 SQL views + 1 MCP tool" describes `llm`-logs-plus-grep. It throws away the three things that are already built, already differentiating, and *cheap to keep*:

- **material_origin** (the authoredness axis) — this is the entire reason cost/user-word accounting is honest. Grep can't reconstruct it. It's not over-engineering; it's the keystone.
- **The MCP continuity surface** (~130 tools, but the load-bearing ~10). This, not the CLI, is where the realized value lives. A single search tool is not the product; `save_annotation` + `get_resume_brief` + recall packs are.
- **Content-hash idempotency + the split-tier durability model.** This is what lets the archive be rewritten/rebuilt without losing the irreplaceable `user.db`. Removing it to "simplify" would trade away the one guarantee that makes durable agent-memory trustworthy.

**So the honest minimal core is not the red-team's, but it is much smaller than the backlog:**

> **Core = { split-tier store + content-hash idempotency + material_origin + FTS + the ~10 real MCP tools (search/get/save_annotation/get_resume_brief/recall) + honest cost rollups + the daemon that owns writes }.** Everything already exists. The core's remaining work is *hardening and activation*, not new subsystems.

The red-team's real, correct signal: **~70% of the P1/P2 backlog is orthogonal to that core and should be deferred or cut** — the insight museum (16-measure catalog, gap-taxonomy, epidemiology), result-set algebra, content-variants/translation epic, the RL/eval-environment moonshots, CIF/federation, and most of the temporal-navigation UI. These are *plausible* but they compound build-cost and honesty-surface without moving the two realized-value axes.

## 2. Top specced subsystems ranked by (operator-value ÷ cost + risk)

Ranked. Convergent designs cited by theme number from the master doc.

| # | Subsystem | Value/cost verdict | Anchor |
|---|-----------|-------------------|--------|
| 1 | **Recursive-safety gate** (agent assertions default CANDIDATE + `inject:false`; closed-loop laundering quarantine) | **Highest.** Now that the archive auto-captures its own R&D chats and feeds them back into context, this is load-bearing *safety*, not a feature. Small (reuses pathology candidate→judge lifecycle + `TopologyEdgeStatus`). Blocks the fabrication class structurally. | themes 5, 9; A18 |
| 2 | **Activation layer** (`polylogue install` one-command hooks + `doctor` liveness; assertions-as-injected-continuity; adoption measured *from* the archive) | **Highest ÷ cost.** The core's value is zero if agents don't use it. Cheap wiring, directly amplifies realized value. | 3gd, d1y, pj8; W2-Act |
| 3 | **Confirmed bug-fix track** (§4) | High, mandatory. These are correctness/security holes in the *core*, not the periphery. | 1xc.11, a7xr.1, 4ts.6, f2qv.5 |
| 4 | **Construct-validity as a gate** (insight-over-zero-rows must FAIL; "unverified candidate" as a distinct render mode; no-regex-over-prose) | High. This *is* the brand ("every number resolves to bytes") and it protects the cost-visibility value. Implement as a CI property test + a render type, not a museum. | themes 2, 12; 9e5 |
| 5 | **Cost-per-outcome join** (`terminal_state × session_costs` → "$ per successful session") | High, low cost — one materializer over existing keystones. The single sharpest "so what" and it rides the *already-realized* cost axis. | theme 4; 9l5.1 |
| 6 | **t46 "contracts own surfaces"** (thin CLI/daemon through one facade; delete parallel `_do_archive_facets` etc.) | High. Prerequisite hygiene: the CLI reaches into substrate 45× vs 18× via API. Every future surface change costs double until fixed. But it's *refactor*, not value — do it as enabling debt, not a headline. | t46.2–.6; swarm B7/B8 |
| 7 | **Content-hash citation anchors** (`session:message:block`, survives re-ingest/fork-shift) | High-value *if* the report/citation UX ships; otherwise latent. It's the atom three lanes converged on. Medium cost. Gate behind whether the operator wants the "evidence cockpit." | themes 1, 3; A9 |
| 8 | **Blue-green index rebuilds** (b5l) | Medium-high. Operational necessity for a derived-tier-rebuild model on a live 38GB archive; without it, schema bumps mean downtime. | b5l; 8jg9 |
| 9 | **Queries/findings-as-objects** (`query:<hash>` ObjectRef, `AssertionKind.FINDING`, findings-as-tests) | Medium. Genuinely elegant and convergent, but it's *infrastructure for the museum*. Defer until a finding actually needs to be a re-runnable test. | themes 1, 8; A2/A17 |
| 10 | **Attachment/blob honesty** (classify 39,586 missing blobs; preserve capture bytes) | Medium. Honesty-critical (the archive currently references bytes it doesn't have) but scoped: the *classification + honest "unfetched" render* matters; wholesale byte-preservation is expensive. Do the honesty half, defer the storage half. | 83u.3/.4 |
| 11 | **Corpus-compaction pack** (`find <q> | compact`) | Medium. The R&D-flywheel enabler (packages archive sessions for the next GPT-pro lane). Value depends on the flywheel being real; medium cost. | theme 6; A11 |
| 12 | **Measure catalog / 5-tuple algebra** | Low-medium. Intellectually the strongest design in the set, but it's the museum. One base measure (cost-per-outcome, #5) delivers 80% of its value; the cartesian product is speculative. **Build the registry contract, not 16 measures.** | themes 7; W2-M, A16 |
| 13 | **Result-set algebra / set-ops** (union/intersect/except) | Low. Orthogonal to both value axes. Elegant, unmotivated by a live operator need. | fnm.13; A2 |
| 14 | **Content-variants / translation epic** (arso, 0v9p, rlsb, d4zk) | Low. A whole epic (6 beads) for translation/OCR/summary variants. No evidence of realized demand; large new durable-tier surface + honesty burden. **Defer indefinitely.** | 4smp |
| 15 | **RL/eval-environment + federation + CIF moonshots** | Lowest (for now). Genuinely exciting, genuinely a different product. Cut from the critical path entirely. | H, D; A12, D7 |

## 3. Dependency-ordered sequence

**Weeks 1–2 — Make the core safe and used (the actual 80%).**
1. **Recursive-safety gate** (#1) — before any more auto-capture feeds context. Blocking prerequisite for the flywheel.
2. **Bug-fix track** (§4) — land the four confirmed correctness/security fixes.
3. **Activation layer** (#2) — `polylogue install` + `doctor` + assertions-as-continuity preamble. This is what converts "built" into "realized."

**Weeks 3–4 — Harden and prove honesty.**
4. **t46 contract thinning** (#6) — enabling debt; unblocks clean surface changes. Do it now while the surface is small.
5. **Construct-validity gate** (#4) + **cost-per-outcome** (#5) — one CI property test, one materializer, one "unverified candidate" render mode. Ships the brand.
6. **Blue-green rebuilds** (#8) — so the derived-tier model is operationally honest.
7. **Attachment honesty half** (#10) — render "unfetched/stale" truthfully; classify the 39k missing blobs. Defer byte-preservation.

**Next (weeks 5–8, only if the above lands clean).**
8. Content-hash citation anchors (#7) + corpus-compaction pack (#11) — *if* the operator commits to the evidence-cockpit/flywheel direction.
9. Queries/findings-as-objects (#9) + the measure *registry contract* (not the 16 measures).

**CUT or defer indefinitely (state it plainly to the operator):**
- **Content-variants/translation epic (4smp + 6 children)** — cut. No realized demand; large durable-tier + honesty cost.
- **Result-set set-algebra (fnm.13)** — defer; unmotivated.
- **The 16-measure museum** — build only the registry contract + the one measure with a live consumer (cost-per-outcome). Let the rest be pull-based.
- **RL/eval-env, CIF/federation, personal-model distillation, temporal-navigation UI (year-heatmap/scrubber/replay), Cursor/Zed/Aider parsers** — cut from the critical path. These are a *second product*; revisit after the core is dogfood-proven.
- **External-legibility polish (3tl.11/.12/.13, agent-forensics regen)** — worthwhile but not critical-path; batch opportunistically.

## 4. Confirmed red-team bug-fix track (own priority lane, lands week 1–2)

All four are real and already filed; treat as a single hardening PR wave. Evidence from the deep-audit + my spot-checks:

- **Convergence freshness probes fail-closed to 'converged'** (`1xc.11`) — `convergence_stages.py` fts check (105–106) returns `False`, `check_many` (161–162) returns `set()`, insights (342, 412) same; a probe *error* silently marks the tier converged → **auto-convergence permanently suspends with no log.** Highest-severity: it silently stops the daemon doing its job. Fix: fail-open + loud log; distinct from the deliberate `false_means_pending` path.
- **sqlite3 connection leaks** (`a7xr.1`) — `with sqlite3.connect() as conn` commits but never closes at ~9 confirmed sites (envelope:591 ×3, user_state_resolver:59/67/91, api/archive:2931/4626, decode:309, repair:112, demo/seed:82). `otlp_correlation:116` already shows the fix pattern. Sweep them.
- **Cost double-count** — `_merge_duplicate_parsed_sessions` (dispatch.py:414–418) *sums* `reported_duration_ms` across merged same-session fragments; a fragment reporting session-total (not per-fragment) double-counts. This is the *live residual* of the Codex-inflation class (the 7.69× bug was fixed in `3938bc6c2`; this is the same failure mode in the merge path). Verify against `claude.parse_code_stream`, then gate. Also file `f2qv.5` (version-gate provider-usage projection so it self-heals like `session_profiles`).
- **Lineage truncation** (`4ts.6`) — `message_query_reads.py`: `_MAX_LINEAGE_DEPTH=64` silently truncates deep acompact chains, and the dangling-branch-point fallback returns tail-only (drops the shared prefix). **No completeness signal in the read envelope** → the archive lies by omission. Fix: surface a `truncated`/`completeness` flag; this rides the same honesty gate as #4 in §2.
- **DNS-rebinding** — the daemon binds `127.0.0.1` and gates non-localhost behind a token (`cli.py:862`), but `http.py:1261` exposes the **web shell as the one unauthenticated endpoint ("localhost only")** with no `Host`-header allowlist. A DNS-rebind attack reaches it from a browser. Fix: validate the `Host` header against an allowlist on the unauthenticated route. Small, security-critical, not yet a bead — **file it.**
- **Dead exact-cost path** — worth a bead: `archive.py:6578` computes `status = "exact" if cost_provenance == "exact"` and a `confidence=1.0` branch; confirm the `"exact"` provenance is ever actually produced upstream, or it's a dead confidence-inflating branch. (Lower severity; verify before fixing.)

## 5. The three decisions only the operator can make

1. **Is the R&D flywheel a real product commitment, or a nice-to-have?** Nearly everything in "Next" (citation anchors, compaction pack, findings-as-objects, self-capture cohorts) is justified *only* by the flywheel — "your history feeds the next GPT-pro lane which feeds beads which feed coding agents." If yes, weeks 5–8 are core; if it's aspiration, cut all of it and stop at week 4. **The recursive-safety gate is a hard prerequisite either way** — do not enable more self-capture-into-context without it.

2. **Museum vs instrument: does Polylogue ship a finding *catalog* or a finding *front door*?** The master doc's own strongest self-critique (theme "hero-finding discipline"): one robust, surprising, clickable number as the spearhead vs a 16-measure catalog with no front door. This decides whether the measure algebra is core (#12) or cut. My recommendation is *instrument* — one hero finding (cost-per-outcome, or replay-duplication %, or the subscription-vs-API gap) + the honesty gate — but only the operator owns the positioning.

3. **Cut the content-variants/translation epic (4smp), or is there a real personal need?** It's a full P1 epic (6 beads, new durable-tier substrate, large honesty surface) with, as far as the evidence shows, no realized operator demand — "the operator's oldest itch" (audio transcripts) is cited, but translation/OCR variants are not. I recommend cutting it from the critical path; but if there's a genuine multilingual/OCR workflow I can't see from the archive, that changes the ranking materially.

---

**Bottom line:** the realized product is *safe, activated agent-memory + honest cost visibility over a durable content-hashed store*. The critical path is ~4 weeks of **hardening (bugs) + safety (recursive gate) + activation + honesty gate**, riding entirely on subsystems that already exist. The elegant convergent designs (measure algebra, findings-as-objects, variants, result-set algebra, RL-env) are real engineering but are *periphery to the two value axes* — build the registry contracts, defer the catalogs, and cut the second product until the core is dogfood-proven.

---

## [a93927f379dd8501e] Grounded on `polylogue-fs1` (Hermes → canonical evidence → forensics/eval export; Atropos 

Grounded on `polylogue-fs1` (Hermes → canonical evidence → forensics/eval export; Atropos JSONL prototype existed; NeMo ATOF/ATIF) and the raised frontier (replay/reproduction, recipe library, queries-as-objects). Here's the moonshot zone.

- **Polylogue as a local RL/eval environment ("your sessions are the eval set")** — every archived session is a labeled trajectory (task, tool-calls, outcome, cost, human corrections already sit in `user.db` as `AssertionKind.CORRECTION`). Export to Atropos/verifiers/NeMo so you can score a candidate model against *your* real work, not synthetic benchmarks — first step: one `polylogue eval export --format atropos` command that lowers N sessions (prompt-prefix → assistant-turn → reward from `tool_result_is_error`/exit-code + correction presence) to JSONL; reuse the fs1 Atropos round-trip. — extends polylogue-fs1, else NEW `epic/eval-env`
- **Reward model from your corrections** — the `assertions` correction/judgment/pathology rows are a standing human-preference dataset over agent behavior. Train a tiny local scorer (or few-shot judge) that predicts "would Sinity flag this turn" — first step: `analyze` recipe that emits (turn_text, was_corrected) pairs + a baseline logistic/embedding classifier, report AUC vs held-out sessions. — NEW `epic/eval-env`
- **Session replay/reproduction harness (re-run tool-calls vs the current repo)** — take a Claude Code session, replay its shell/edit tool-calls against a fresh checkout at the session's git SHA, diff produced state vs recorded results → measure model determinism, catch "worked then, breaks now" regressions, generate verifiable RL rewards. First step: a read-only *dry* replayer that extracts the ordered tool-call list + cwd + git SHA from one session and prints an executable plan (no execution yet). — NEW `epic/replay`, links frontier replay item
- **Counterfactual re-run ("what if model X had done this task")** — freeze the human turns + initial repo state of a real session, re-drive with a different model/config, compare cost/turns/outcome against the actual recorded run. Turns the archive into an A/B substrate for model selection. First step: extract "human-authored turns only" transcript (already have `material_origin` filter) as a replayable prompt script + a `compare_sessions`-style diff over the two runs. — NEW `epic/replay`
- **Standing "resident intellect" agent that mines the archive nightly** — a scheduled agent whose *only* context is polylogue MCP + assertions; it surfaces recurring pathologies, unfinished threads (`find_abandoned_sessions`/`find_stuck_sessions` already exist), cross-session contradictions, and drops findings as `note`/`blocker` assertions. First step: a `schedule`d `claude -p` job with a fixed recall-pack prompt that writes ≤5 findings/day via `save_annotation`. — NEW `epic/resident`
- **Personal-model distillation substrate** — the archive is a high-signal SFT corpus of *how you actually collaborate* (your phrasings, your accept/reject patterns, your domain). Curate a "golden turns" subset (marked + uncorrected + non-error) as distillation data for a local model that mimics your assistant-preferences. First step: a `select`-backed export of turns tagged `golden` with material-origin + error gating, count tokens, report subset composition. — NEW `epic/personal-model`
- **The archive as agents' cross-project long-term memory (MCP "recall the last time we hit this")** — polylogue MCP is *already* the continuity surface; make it the default long-term memory for every project's agents via a single `recall(task_hint)` tool that returns the most similar prior sessions + their corrections/lessons across ALL repos, not just cwd. First step: a thin MCP tool wrapping `find_similar_sessions` + `get_pathologies` keyed by embedding of the current task description. — extends MCP surface, NEW `epic/cross-project-memory`
- **Reproducibility/eval "spec cards" per session** — auto-derive a portable task-spec from each session (initial repo SHA, human intent, acceptance signal, final diff) so any session can become a benchmark item shareable without transcript leakage. First step: a `spec-card` insight (registry descriptor) that emits {intent, start_sha, tools_used, outcome_signal} JSON for one session. — NEW, feeds `epic/eval-env`
- **Privacy-preserving cross-archive federation (compare your patterns to a peer's without sharing transcripts)** — exchange only *derived statistics/embeddings/pathology-rates*, not content, so two operators can benchmark model behavior or share "recipes that work" federated. First step: define a `federation-digest` export = aggregate insight vectors (tool-usage distribution, pathology rates, cost-per-outcome) with zero raw text, plus a diff tool between two digests. — NEW `epic/federation`
- **Analysis-recipe library as first-class objects (queries + insight pipelines you can name, share, run)** — the DSL already lowers `sessions where … | group by … | count`; promote saved queries + multi-step analysis chains to versioned, shareable "recipes" with parameters, so "find my stuck refactors" is an artifact not a re-typed query. First step: extend `save_saved_view` to store a parameterized DSL pipeline + a `run-recipe <name> --param k=v` verb. — extends frontier recipe-library item, NEW `epic/recipes`
- **Pathology-driven auto-guardrails (mine failures → emit context-injected caveats)** — you already detect pathologies + store injectable assertions (`context_policy_json`). Close the loop: recurring failure modes become auto-generated `caveat` assertions that inject into future agent preambles ("last 3 times you edited convergence_stages.py you broke X"). First step: a job that clusters pathology rows by touched-file and proposes one caveat assertion per cluster (operator-gated). — NEW `epic/resident`, links `#1498 cascade` doc
- **"Time-machine" model-drift observatory** — the archive spans providers/models/dates; treat it as a longitudinal instrument to *measure how models changed* on stable task shapes (same repo, similar intent, different month/model). First step: a `workflow_shape_distribution`-style cohort query bucketed by (model, month) over sessions matching a fixed intent embedding, plotting cost/turns/error-rate drift. — NEW `epic/observatory`
- **Verifiable-reward tasks auto-mined from your CI-passing sessions** — sessions that ended with a green `devtools verify`/tests-pass tool-result are gold RL tasks with a *checkable* reward (re-run the verify command). Harvest them into a verifiers-compatible env. First step: query for sessions whose last tool-result is a passing test/verify command (structure-read from `tool_result_exit_code=0`) and emit (task, verify_cmd, start_sha) triples. — NEW `epic/eval-env`, depends `epic/replay`
- **Self-improving CLAUDE.md / skill synthesis from lived corrections** — MEMORY already notes "assertions > CLAUDE.md"; make it generative: the resident agent proposes CLAUDE.md/skill edits derived from repeated corrections and pathologies, as PRs against the dotfiles. First step: weekly digest that groups CORRECTION assertions by theme and drafts one candidate memory/skill snippet (operator reviews, never auto-commits). — NEW `epic/resident`
- **Polylogue as the substrate for a personal AI that knows your whole collaboration history** — the umbrella: a single agent front-end whose grounding is the full archive + assertions + Lynchpin cross-source telemetry, answering "what did we decide / try / abandon about X" across years and providers. First step: a curated "resident" MCP profile bundling polylogue recall + assertion-claims + Lynchpin, and one canonical prompt that always leads with a recall-pack. — NEW `epic/resident`
- **Trajectory quality index (per-session "collaboration health" score) as a training/eval signal** — combine latency profile, correction density, abandonment, tool-error rate, phase thrash into one normalized score; use it both as an RL reward shaping term and as a personal dashboard of when *you+model* work well. First step: an insight descriptor composing existing `session_latency_profile`/`get_pathologies`/`session_phases` into a single 0–1 score with documented weights. — NEW, feeds `epic/eval-env` + `epic/observatory`

---

GPT-pro prompt stubs:

- **[DR]** "Survey the current (2026) landscape of RL/eval environments and trajectory formats for coding agents — Atropos (Nous), verifiers, NeMo Relay ATOF/ATIF, OpenAI evals, terminal-bench/SWE-bench-style harnesses. For each: input schema, reward model, whether human-correction signals are first-class, and how a local personal corpus of ~10k real coding sessions (tool-calls, exit codes, human corrections, git SHAs) could be lowered into it with minimal fabrication. Recommend the single most leverageable export target for a solo operator's archive and the exact JSONL shape."
- **[DR]** "Map the state of personal-model training/distillation from one's own AI-collaboration history: what corpus size/quality is actually needed for useful SFT vs preference-tuning (DPO/reward-model) of a local model; how practitioners gate for data quality; privacy/federation techniques for comparing behavioral statistics across archives without sharing transcripts. Give a concrete minimal pipeline a single person could run on ~40GB of session data."
- **[A]** "Critique this reframe: turning a passive AI-session archive into a session *replay/reproduction* harness that re-runs recorded tool-calls against the repo at the recorded git SHA to (a) measure model determinism, (b) generate verifiable RL rewards, (c) A/B new models on real past tasks. Enumerate the hard correctness/safety/nondeterminism problems (side effects, network, time, filesystem state), which session types are safely replayable vs not, and the smallest trustworthy first slice."

---

## [a164e273173fdbbe8] - Provenance stanza as a queryable table, not prose — materialize the five-part finding st

- Provenance stanza as a queryable table, not prose — materialize the five-part finding stanza (cursor-id/position at measurement, measure+DSL/code version, git SHA, sample-frame predicate, run-date) into an `index.db` `finding_provenance` table so `find` can filter "findings whose sample-frame predicate no longer matches current rows" and auto-flag staleness on re-ingest, rather than trusting a markdown header nobody re-reads — makes construct-validity a live query surface, not a doc convention. — polylogue-cpf / polylogue-3tl.4

- Sample-frame drift detector: store the exact population query behind every published finding and re-run its `COUNT` on each converger pass; when the denominator shifts >X% the finding's page auto-stamps "population changed 412→1,088 since measurement — re-run before citing" — catches the single most common construct-validity lie (a number computed on a frame that silently grew) without a human noticing. — NEW (child of polylogue-3tl.4)

- Trust-class as a typed column on injected context, refusing to blend rungs — every row that can reach an agent's context window carries `trust_class ∈ {OPERATOR, SYSTEM, QUOTED}`; the context compiler physically cannot concatenate a QUOTED (session-content, attacker-controllable) span into the same block as an OPERATOR instruction without a visible fence + provenance ref, so prompt-injection from archived transcripts can't impersonate operator intent. — polylogue-cpf (37t.11)

- Deny-lexicon tripwire fixture shipped as a standing test, not a one-time audit — a corpus of known injection strings ("ignore previous instructions", "you are now", tool-call spoofs) that must round-trip through `compile_context`/`compose_context_preamble` and come out fenced+attributed as QUOTED; the test fails if any lands unfenced — turns injected-context-trust from doctrine into a regression gate. — polylogue-cpf

- Secret-scan at the READ boundary (fold into read, per operator), not a separate sanitize flow — `read`/`get_messages` runs an entropy+gitleaks-class detector on emit and returns spans as `secret_candidate` overlays with a "reveal" affordance, so the archive keeps everything durably but never re-surfaces a pasted API key by accident; detection is a read-time projection, redaction-on-disk stays the separate excision op. — polylogue-kwsb / polylogue-27m

- Secret candidates go to the judgment queue as assertions, never auto-redacted — detection emits `AssertionKind.secret_candidate` rows (span ref + detector + confidence), grouped in the judge surface; acceptance triggers excision, rejection records a false-positive that suppresses the pattern for that span — respects the judgment gate and builds a labeled corpus that tunes the detector over time. — polylogue-27m

- Verifiable excision with a content-hash tombstone: prove removal without a re-import resurrection — excise replaces the payload with a hash-of-removed marker, records the session's OLD content_hash on the tombstone, and recomputes a new one, so idempotency can't re-ingest the removed span AND an auditor can later prove "exactly this bytes-hash was removed at this time for this reason" without the archive retaining the secret. — polylogue-27m

- One shared mutation-audit contract across excise / reset / delete_session / MCP admin — every destructive path emits the same `ops.db` audit row (actor, ref, dry-run-diff hash, reason, --yes provenance) and every path must produce a dry-run diff before mutation; fixes the reset.py:260-277 tombstone-before-preview bug class structurally instead of per-command — "the archive can forget, and every forget is itself remembered." — polylogue-kwsb / polylogue-jnj.5

- Degraded-mode ladder surfaced in every read payload's envelope, degrade-loudly-once — each read/insight response carries a `degradation_rung` (full / partial-index / embeddings-stale / evidence-only / …) so a caller never mistakes "no results because empty" for "no results because embeddings 3 days behind"; the EVIDENCE-ONLY floor is sacred: even fully degraded, raw-source reads still answer. — polylogue-cpf

- Claim-vs-evidence as a standing invariant test over insight OUTPUT, not just code — the existing `test_claim_vs_evidence` idea generalized: every `InsightType` must declare its evidence accessor, and a property test asserts no rendered insight string contains a quantitative claim whose backing rows are empty/NULL (the recovery-digest "PR #123 merged" fabrication class) — construct-validity enforced at the registry boundary. — polylogue-9e5 (extends insight_rigor_audit)

- "Unverified candidate" as a first-class render mode distinct from "fact" — text-mined events (regex over prose, no structural authorship) render in a visually + schematically separate `candidates` block that can never be promoted to a claim without a structural or operator-judged evidence ref; codifies the 2026-06-29 recovery-report fix into the type system so no future insight can quietly launder mined prose into asserted history. — polylogue-9e5

- Confidence + evidence-ref REQUIRED on every assertion at write time, refused otherwise — extend `user_write.py` so no assertion (correction/judgment/pathology/…) persists without {schema-version, prompt-or-method id, model id if LLM-authored, evidence_ref, confidence}; a null in any field is a write refusal, making the assertion table self-describing for a future auditor who asks "where did this judgment come from?" — turns provenance-completeness from aspiration into a NOT NULL. — polylogue-cpf / polylogue-9e5.1

- Semantic-version-bump invalidation cascade for findings and assertions — when a measure's DSL/code version bumps (breaking), every finding/assertion computed under the old version flips to `stale` and its published page shows a diff banner; re-runs supersede rather than overwrite (append-only lineage of the same slug) so an external citer can see the number moved and why. — polylogue-cpf / polylogue-3tl.4

- Adversarial self-audit shipped as `polylogue audit --adversarial <insight>` — a bounded read-only lane that takes any insight/finding and generates the strongest counter-frame (alternative sample-frame, confounded denominator, survivorship, provider-token vs origin collapse) as structured `caveat` assertions attached to the finding — the adversarial-loop skill, productized: the archive argues against its own numbers before an operator cites them. — polylogue-9e5

- Injected-context provenance receipt: every agent-facing context bundle ships a manifest of what it contains and why — `build_context_image`/`compile_context` attach a signed-ish manifest (source refs, trust classes, assertion ids, freshness per component) so a downstream agent (or auditor) can reconstruct exactly which archive rows shaped a decision — makes the continuity surface auditable instead of an opaque blob, and lets a paranoid caller drop QUOTED components. — polylogue-cpf (37t.11)

- Forget-audit reconciliation check: prove no derived tier retained excised content — a converger invariant that, after any excision, scans index/FTS/embeddings/blob tiers for the recorded removed-hash and fails loudly if it reappears (rebuild race, stale blob lease), so "verifiable deletion" is continuously verified, not asserted once at excise time — deletion durability gets the same continuous-check rigor lineage got. — polylogue-kwsb / polylogue-27m

- Construct-validity metadata on the origin projection itself: flag non-injective collapses at the boundary — since GEMINI + DRIVE → AISTUDIO_DRIVE is lossy, any count grouped by `origin` carries a `lossy_grouping` marker naming which provider tokens were merged, so a cost/usage number can't silently conflate two runtimes under one origin label — honesty about the archive's own vocabulary compromises. — polylogue-9e5 (relates provider→origin)

---

GPT-pro prompt stubs:

- [A] "Design a `trust_class` type system (OPERATOR/SYSTEM/QUOTED) for a context-compiler that assembles LLM context from an archive of adversarial transcripts. Specify: the invariant that prevents QUOTED (attacker-controllable) spans from being read as instructions, the fencing/attribution format, the refusal semantics when a bundle would mix classes, and a test corpus design that proves known injection strings survive as inert QUOTED. Give concrete Python protocol signatures."

- [DR] "Survey verifiable-deletion / right-to-forget techniques in append-only, content-addressed local stores (tombstoning, hash-of-removed markers, crypto-shredding, Merkle-consistency proofs). For each: how it interacts with content-hash idempotency and derived-index rebuilds, what an auditor can prove afterward without retaining the deleted bytes, and failure modes where derived tiers silently resurrect removed content. Cite systems that do this well."

- [DR] "Research 'construct validity as a product feature' in analytics/observability tools: how do the best-in-class systems expose sample-frame, denominator provenance, and staleness so a number can't be silently miscited? Compare approaches (dbt tests, Great Expectations, metric-store semantic layers, incident-postmortem tooling) and extract a concrete design for auto-flagging findings whose population predicate has drifted since measurement."

---

## [a85d12b6f50927177] THE DEMO / PROOF ENGINE — deep mechanics (grounded in `polylogue/demo/{seed,verify,constru

THE DEMO / PROOF ENGINE — deep mechanics (grounded in `polylogue/demo/{seed,verify,constructs,tour}.py`, `devtools/claim_vs_evidence.py`, `docs/examples/demo-tour/`, `insights/portfolio.py`)

- Executable-prompt-file demo format (`*.polydemo` = frontmatter budget + ordered CLI steps + expected-construct assertions) that a stranger `uvx polylogue demo run FILE` executes to an identical artifact — promote `demo/tour.py`'s hardcoded step list + `FIRST_RESULT_BUDGET_S/FULL_TOUR_BUDGET_S` into a declarative, shippable file so the demo IS the spec, not code — NEW
- Content-addressed "citable finding" object: `finding_id = sha256(claim_text + sorted(evidence_refs) + corpus_datasheet_hash)` minted at report time, embedded in `report.json` and every rendered permalink — makes a finding a durable noun a cold reader can re-resolve, extending the `EvidenceRef` model already threaded through `portfolio.py`/`claim_vs_evidence.py` — NEW
- Round-trip evidence-ref resolver as a hard gate: `demo verify` walks every `EvidenceRef` in the emitted report back through `resolve_ref` and fails if any dangles — closes the gap where a report can cite `session:msg:block` coordinates that no longer exist after a corpus rebuild — NEW
- Anti-demo manifest (`REFUSALS.md` generated beside `COLD_READER_GATE.md`): the demo explicitly enumerates claims polylogue *won't* make on the demo corpus (no cost-per-outcome without ≥N labeled outcomes, no pathology rate below sample floor, no provider→origin projection where GEMINI/DRIVE collapse is lossy) — turns the `_BENIGN_RECOVERY_TOOLS` vs `_CONSEQUENTIAL_TOOLS` "methodology split, not a truth claim" comment into a first-class shipped artifact — NEW
- Construct-validity gate as pre-render tripwire: run `evaluate_demo_constructs` (the `DemoConstruct.sql`/`minimum` coverage table) *before* any narrative renders, and hard-abort the tour if any declared construct is under-observed — a demo that would fabricate a finding on a corpus missing its supporting construct cannot emit — extends `demo/constructs.py` — NEW
- Demo-as-CI-test / stale-finding tripwire: register each shipped `docs/examples/demo-tour/report.json` finding as a `render all --check` surface — the devloop fails when a re-run produces a different `finding_id`, so a silently drifted claim (schema change, metric bugfix) breaks the build instead of rotting in a gif — hooks into existing `render --check` "out of sync" gate — NEW
- Corpus datasheet as the reproducibility contract: extend `.cache/demo-corpus-datasheet/` into a signed `datasheet.json` (session/message/construct counts + tier `user_version`s + seed) that every report embeds, so "identical artifact for a stranger" is *checked* — the real-38GB report carries its own private datasheet hash, letting the same pipeline prove reproducibility on synthetic data and honesty on private data — NEW
- Two-corpus parity harness: one `claim_vs_evidence.build_report` invocation runnable against demo-corpus (public, committable golden) and the live archive, with a `--parity` mode asserting the *pipeline* (not the numbers) is byte-identical — proves the privacy-free demo exercises the exact code path that runs on 38GB, killing "the demo is a rigged toy" objection — NEW
- Cold-reader gate escalation: promote `COLD_READER_GATE.md` from prose to executable checklist — each gate line is a `pytest` node in `test_claim_vs_evidence.py` (evidence resolves, calibration κ ≥ floor, sample stratification deterministic under seed) so "a cold reader can trust this" is machine-verified, not asserted — extends existing calibration-metrics code (`_calibration_metrics`, seed=7) — NEW
- Inter-rater calibration baked into the demo: ship `ack-marker-calibration.labels.csv` as the golden human-label set and have `demo verify` recompute Cohen's κ between the classifier and the frozen labels, failing if agreement regresses — turns the claim-vs-evidence acknowledgement classifier into a self-auditing, drift-detecting demo rather than a one-shot report — NEW (build on `.agent/demos/claim-vs-evidence/`)
- The Fable "iron fist" demo, done rigorously: a `delegation.polydemo` whose comedic hook is a parent agent's grandiose delegation prose, whose payload is the serious `delegations where … | group by subagent_model_family | count` DSL + delegation-yield metric (child cost/tokens/wallclock vs parent terminal success) — the joke is the framing, the artifact is a citable subagent-ROI finding — NEW (generalizes Fable demo; ties to `rnd-brainstorm` A4 delegation substrate)
- Leak-proof gif provenance: `verify.py` already rejects raw `source_path` leaks — extend the same reject-list to the *rendered* surfaces (VHS `recording.tape`, `transcript.txt`, `demo-tour.gif` frames via OCR-free text-layer check) so a demo asset can never ship a private path/title even if a future step prints one — NEW
- `uvx`-proof as a runtime assertion, not a doc: `docs/examples/demo-tour/uvx-proof.md` becomes a live step that shells `uvx --from . polylogue demo verify` in a clean tmp XDG and diffs the artifact hash against the committed golden — proves "zero-install stranger reproduces it" on every render instead of trusting a stale screenshot — NEW
- Finding → permalink → PR-comment loop: a citable `finding_id` resolves to a stable local `polylogue finding show <id>` view AND a copy-pasteable evidence block; wire it so a demo finding cited in an issue/PR body is a durable reference a reviewer can re-run, not a number that decays — the "findings-as-objects" convergent theme applied to the proof surface — NEW
- Budget-honest demo telemetry: record actual per-step wall-clock against `FIRST_RESULT_BUDGET_S`/`FULL_TOUR_BUDGET_S` into `report.json` and fail the tour if first-result exceeds budget on reference hardware — a demo that quietly got slow (a perf regression) breaks loudly, and the "30s to first citable result" claim stays true or the build stops — extends `demo/tour.py` — NEW
- Portfolio-as-demo hardening: run `insights/portfolio.py` (#2437) over the demo corpus as a committed golden narrative so the sanitizer + pathology + context-loss aggregation path is proven privacy-free before it ever touches the 38GB archive — the corpus-scope sibling of the per-finding gate, catching sanitizer regressions in the devloop — NEW (build on #2437)

GPT-pro prompt stubs:

- [A] "Design the `*.polydemo` executable-prompt-file format for Polylogue: a single committable file a stranger runs via `uvx polylogue demo run FILE` to reproduce a byte-identical citable artifact. Given the existing `demo/tour.py` (hardcoded steps, `FIRST_RESULT_BUDGET_S=30`, `FULL_TOUR_BUDGET_S=420`), `demo/constructs.py` (`DemoConstruct` sql+minimum coverage), and `demo/verify.py` (rejects raw source-path leaks), specify the frontmatter schema (budgets, corpus datasheet hash, declared constructs), the step grammar (CLI verb + expected `finding_id`/construct assertions), the failure taxonomy (construct under-observed, dangling EvidenceRef, budget exceeded, artifact-hash drift), and how the same file doubles as a `render all --check` CI surface. Show the format's collision with the query-first CLI and how a step references a query without re-encoding argv ordering rules."

- [DR] "Survey how reproducible-research / claims-with-evidence systems (datasheets for datasets, model cards, RO-Crate, DVC/`dvc repro`, Snakemake report, Jupyter Book executable-check, Quarto freeze, in-toto/SLSA provenance, ClaimReview schema, registered-reports/pre-registration, Cohen's/Fleiss κ inter-rater protocols) establish that a *stranger reproduces an identical, cited artifact* and that a claim can't outrun its evidence. Extract concrete mechanisms Polylogue can adopt for: content-addressed citable findings, a corpus datasheet as reproducibility contract, cold-reader gates, calibration-as-CI, and an explicit anti-claim manifest. For each, give the mechanism, the failure it prevents, and a minimal Python/SQLite-first adaptation."

- [A] "Critique Polylogue's 'demo-as-CI-test' and 'construct-validity gate' proposals as a rigor system: could a passing demo still mislead a cold reader? Adversarially find the holes — construct coverage satisfied but semantically wrong; `finding_id` stable yet the underlying metric is biased; calibration κ high on an easy stratified sample but low on hard cases; parity between demo-corpus and 38GB proving code-path identity while hiding numeric distortion. Then propose the smallest set of additional gates (sample floors, hard-case oversampling, refusal manifest coverage checks, evidence-ref round-trip) that would make a fabricated or drifted finding provably fail the devloop, and rank them by cost-to-implement vs objections-killed."

---

## [a8b507c2e044a3768] SEARCH RELEVANCE, RANKING QUALITY & EXPLAINABILITY

Grounded in `docs/search.md`, `storage/search_providers/hybrid.py` (equal-weight RRF k=60, `fts_limit=limit*3`), `sqlite/queries/sessions_search.py` (flat bm25, snippet), and `archive/query/miss_diagnostics.py`. Findings: fusion is unweighted, no lineage-dedup in `hybrid_sessions`, `score_components` already carries per-lane RRF decomposition (#1267), miss-diagnostics exists but has no did-you-mean/relaxation, and there is no relevance-judgment substrate.

**SEARCH RELEVANCE, RANKING QUALITY & EXPLAINABILITY**

- Lineage-aware result dedup/collapse — forks/resumes/acompaction physically replay parent prefixes, so one logical thread emits N near-identical hits and crowds the top-k; collapse hits sharing a `session_links` root into a single result carrying a `variant_count` + best-anchor, ranked by the strongest member. Reuses the lineage recomposition that reads already do (#2467) — NEW

- `explain search` verb / `--why` flag rendering `score_components` as prose — the RRF decomposition (`text_rrf`+`vector_rrf`, lane ranks) exists in the payload but no surface narrates it; emit "ranked #2 because: matched 'timeout' lexically (rank 1) AND semantically near your phrase (rank 3); recency neutral." Turns #1267 evidence into an operator-legible artifact — builds on citation-anchors + #1267 — NEW

- Relevance-judgment assertions (LtR substrate) — add `AssertionKind.RELEVANCE` rows keyed `(query_fingerprint, target_ref, judgment∈{relevant,irrelevant,best})` written by `mark` on a result; the schema-free TEXT AssertionKind means no user-tier bump. This is the corpus for every tuning/eval idea below — operator marks become durable training signal — builds on findings-as-objects + saved-relevance — NEW

- Query-set relevance eval harness (`devtools lab search-eval`) — a golden set of real queries × judged targets from the RELEVANCE assertions, reporting nDCG@10 / MRR / recall@k per lane and per ranking_policy_version; gate weight/k changes on non-regression. Makes "is search actually good?" a measurable law, not vibes — builds on construct-validity-as-substrate — NEW

- Per-lane RRF weights (weighted fusion) — `reciprocal_rank_fusion` sums `1/(k+rank)` with implicit weight 1.0 per lane; add `w_lane` so lexical-exact can outweigh fuzzy-semantic (or vice-versa) and expose it in `ranking_policy_version`. Tune via the eval harness, not by hand — NEW

- Query-intent classifier before lane routing — a bare term, an id/prefix, a code-symbol, an error string, and a conceptual phrase want different lanes; `auto` currently always resolves to `dialogue`. Route error-strings/symbols to exact-lexical-boosted, conceptual phrases to hybrid, id-shaped tokens to identity — recorded on the envelope as `resolved_intent`. Builds on the #1842 command-floor "signalled intent" work — NEW

- Did-you-mean / query relaxation in `QueryMissDiagnostics` — the miss-diagnostics struct enumerates reasons but suggests no repair; add candidate relaxations (drop the narrowest predicate, widen `since`, spell-correct a rare FTS term against the term dictionary, fall to prefix match) each with the count it *would* yield, so zero-result becomes one-keystroke recovery — extends `miss_diagnostics.py` — NEW

- Field-boosted bm25 (title/first-message/code weighting) — `messages_fts` is single-column flat bm25; split `search_text` provenance so a title or opening-prompt hit outranks a buried tool-log hit via FTS5 column weights `bm25(fts, w_title, w_body, …)`. A match in the question the operator actually asked is worth more than one in a 400-line paste — NEW

- Recency×relevance blend as an explicit, tunable knob — ranking is currently pure relevance; add an age-decay multiplier `score * exp(-λ·age)` with λ surfaced in `ranking_policy` and defaulting to 0 (opt-in), so "what did I try *recently* about X" doesn't return a six-month-old top hit. Blend weight is another eval-harness-tuned parameter — NEW

- Snippet quality upgrade: best-window + query-term-density selection — FTS5 `snippet()` picks the first match window; for long tool-heavy messages pick the highest term-density window and expand to a sentence boundary, and prefer human/assistant prose blocks over runtime_protocol rows via `material_origin`. Snippet is the operator's relevance proxy; a good one halves click-through cost — NEW

- Diversity re-rank (MMR) over top-k after fusion — beyond exact lineage dupes, near-duplicate sessions (same repo, same error) cluster; apply maximal-marginal-relevance using existing embeddings to trade a little relevance for coverage, so page 1 spans distinct problems not ten takes on one. Toggle + λ on the envelope — builds on corpus-compaction — NEW

- `saved_query` + attached judgments = a living relevance regression fixture — the `AssertionKind.saved_query` rows already exist; bind each to its RELEVANCE judgments so a saved query is self-testing: re-running it after any index/weight change diffs the result-set against the last-judged ordering and flags drift. Queries-as-objects that assert their own quality — builds on queries/findings-as-objects — NEW

- Ranking-policy-version drift detector as a convergence stage — the version field is a declared contract; add a stage that snapshots top-k for the eval query set per index rebuild and alarms when ordering shifts without a version bump, catching silent tokenizer/DDL-driven relevance regressions. Recursive-safety gate applied to ranking — builds on recursive-safety — NEW

- Score-calibration layer for cross-query comparability — raw bm25 (negative, query-dependent) and vector distance are non-comparable across queries, so no honest "how confident is this hit" exists; add a per-lane rank-percentile / min-max normalization into `score_components` (clearly labelled derived, not raw) so surfaces can render a calibrated confidence without the "never show bm25 as percent" footgun — builds on construct-validity — NEW

- "No-signal" honesty band on hybrid hits — a hit present in only one lane at a deep rank contributes a tiny `1/(k+rank)` yet still ranks; tag hits below a support threshold (single-lane, rank > N) as `weak_evidence=true` so agents/context-compilers don't cite a rank-58 fluke as a finding. Analytics-one-measure-away: the measure is lane-support count — builds on citation-anchors + construct-validity — NEW

- Highlight/anchor round-trip test — matched_terms must actually appear highlighted in the snippet and the `anchor`/`target_ref` must resolve to a block containing them; a property test over real queries pinning matched_terms ⊆ snippet-tokens ⊆ resolved-block prevents explainability rot — builds on citation-anchors — NEW

**GPT-pro prompt stubs**

- [DR] "Reciprocal Rank Fusion tuning in production hybrid lexical+vector retrieval: survey evidence on choosing k, per-lane weights, and score normalization vs. rank-only fusion; when weighted RRF beats equal-weight; how teams build offline relevance-judgment sets and which metrics (nDCG@k, MRR, recall) actually predict operator-perceived quality on small (<100k doc) personal corpora. Cite papers + real systems."

- [DR] "Result diversification and duplicate-collapse in search over versioned/forked document trees (git-like or conversation-fork lineage where documents share large prefixes): compare MMR, cluster-then-rank, and canonical-representative collapse; how to pick the surfaced representative and how to score a collapsed group. Include failure modes where dedup hides the actually-relevant variant."

- [A] "Design an explainable ranking UX for a hybrid FTS5-bm25 + vector-RRF search where each hit already carries per-lane rank/rrf components. Given this score_components schema {text_rank,text_rrf,vector_rank,vector_rrf}, propose (a) a compact prose 'why ranked here' template, (b) a calibrated cross-query confidence signal that avoids showing raw negative bm25 as a percent, and (c) a 'weak evidence' threshold rule. Output concrete formulas + example renderings."

---

## [a8bcb7cef9ce188ba] INCIDENT / POSTMORTEM / COMPACTION-LIFECYCLE / RESILIENCE — 15 ideas

Grounded in polylogue-gjg (compaction lifecycle), 8jg9 (operational resilience), peo (daemon-death forensics), and the existing postmortem-bundle insight (`insights/postmortem.py`, honest-degradation pattern) + resume-brief/abandoned/stuck MCP tools.

---

**INCIDENT / POSTMORTEM / COMPACTION-LIFECYCLE / RESILIENCE — 15 ideas**

- **Compaction as a first-class archived object** — today acompact leaves only a lineage edge (v12); mint a `CompactionEvent` row (pre-boundary msg id, post-continuation msg id, harness-summary text, trigger=auto/manual, token budget at fire) so "the agent forgot X" becomes a queryable event with a stable ref, not a folklore complaint — polylogue-gjg / NEW
- **Loss-forensics as a registered construct measure, tier=structural** — diff pre-snapshot vs post-compact early context for structurally-extractable items (file paths, tool_id outcomes, marked decisions, refs); emit present-before/absent-after with a construct-validity tier so corpus epidemiology ("median compaction drops 34% of marked decisions") is finding-grade, not vibes — polylogue-gjg (9l5.7) / NEW
- **JSONL-boundary fallback snapshot, always-on** — don't gate the whole lifecycle on PreCompact hook liveness (the hook catalog drifts); the JSONL up to the compaction boundary IS a lossy pre-state — implement + label it `snapshot_source=jsonl-boundary` vs `snapshot_source=precompact-hook` so forensics degrades honestly rather than going dark on hookless runs — polylogue-gjg / NEW
- **Content-addressed compaction snapshots via existing blob substrate** — repeated compactions of a long session share ~all prefix bytes; store the assembled-context snapshot through `blob_refs`/dedup so snapshotting every compaction is near-free, and the lease/GC invariants already protect it — polylogue-gjg + 8jg9.2 / NEW
- **Re-grounding lane keyed to measured loss, not generic recap** — opt-in SessionStart(source=compact) that injects top-K lost-but-later-referenced items as resolve_ref-expandable refs, ~200 token budget, ranked by loss-forensics score; jgp-compliant because volume is bounded by evidence, and it's arm-able as an ExperimentSpec (compact-with vs compact-without) — polylogue-gjg / NEW
- **"What did this agent forget" MCP tool** — a first-class `compaction_forgot(session_id)` that returns the ranked lost-item list with refs; this is the continuity-surface complement to `get_resume_brief` — one is voluntary resumption, the other is involuntary-loss recovery — polylogue-gjg / NEW
- **daemon_lifecycle table + heartbeat as the resilience keystone** — faulthandler + SIGTERM/SIGINT handlers logging thread stacks to run-log AND an ops.db `daemon_lifecycle` row (started/stopped/signal/last_heartbeat) before exit; an atexit sentinel splits clean-stop from vanish — the exit-code-144 ambiguity IS the finding: nothing records why the daemon died — polylogue-peo / existing
- **Heartbeat-backed liveness, not pid-file truth** — bare `status` and `/healthz` report heartbeat AGE; a "running" claim must be backed by a fresh tick (the bug: bare status said "running" while it actually served a rebuild). Web SPA shows "daemon unreachable since T, retrying" banner instead of per-widget "Failed to fetch" — polylogue-peo / existing
- **Postmortem bundle should ingest the incident, not just the session** — extend `compile_postmortem_bundle` to attach `daemon_lifecycle` deaths, convergence_debt strandings, and compaction events overlapping the session window, each as an EvidenceRef; the incident timeline becomes part of the shareable artifact instead of living only in the run-log — 8jg9 + postmortem.py / NEW
- **Crash-recovery convergence-debt reconciliation** — daemon crash mid-convergence must not strand debt: on restart, replay `convergence_debt` + re-check the last in-flight session-scoped stage (check_sessions/execute_sessions already exist for exactly this retry path) and log the recovery as a lifecycle event — polylogue-peo (ties 1xc.3/1xc.4) / existing
- **Restore-drill as a durable artifact + finding** — the quarterly restore drill (4be) should emit a machine-checkable artifact (`devtools workspace restore-drill --json`): backup manifest verified → restored into scratch archive → row/count parity against live durable tiers (source.db, user.db); a failed drill is a P1 finding-object, not a silent gap — polylogue-4be / existing
- **Deploy-safety smoke as one measure away** — `deployment-smoke --json` proves deployed state matches expected schema versions + a fresh heartbeat + non-empty durable tiers; wire it so re-activating prod polylogued fails loud on version skew rather than serving a silently-degraded archive — polylogue-s8q / existing
- **Compaction-loss survives compaction itself (recursive-safety)** — the re-grounding injection and the loss-forensics record must themselves be archived as assertions so a SECOND compaction can re-ground from the FIRST's forensics; without this the OS-vision handoff chain breaks at depth 2 — apply the recursive-safety gate to the lifecycle itself — polylogue-gjg / NEW
- **The controlled-handoff triad, documented as one map** — voluntary handoff (37t.3 reboot-with-refs), involuntary compaction (gjg), cross-session resumption (resume-brief) are the same OS-memory-management story from three trigger-points; render the triad in the 37t epic so incident/compaction/resumption stop being treated as three unrelated features — polylogue-gjg / 37t / existing
- **Corpus-compaction epidemiology view** — aggregate compaction events archive-wide into a rendered table: loss rate by provider (Claude Code acompact vs Codex), by session length bucket, by trigger type; this is the analytics-one-measure-away payoff — the CompactionEvent + loss measure exist, the view is a group-by over them — polylogue-gjg / NEW

---

**GPT-pro prompt stubs**

- **[A]** "Design the `CompactionEvent` schema + loss-forensics diff algorithm for a single-writer SQLite archive of LLM sessions where forks/compactions physically replay the parent prefix (only the divergent tail is stored). Given a pre-compaction assembled-context snapshot (blob) and the post-compaction continuation transcript, specify a deterministic, structurally-grounded diff that classifies each pre-item as retained/lost/transformed for tiers {file-path, tool-outcome, marked-decision, cited-ref}, ranks lost items by later-reference likelihood, and degrades honestly when the snapshot source is a lossy JSONL boundary rather than a true PreCompact payload. Give the index-tier DDL, the pure aggregator signature, and the honest-degradation reason strings."

- **[DR]** "Survey how autonomous-agent frameworks (Claude Code, Codex, LangGraph, AutoGPT-lineage, OpenAI Assistants) currently handle context compaction / summarization-and-discard, and whether ANY of them (a) snapshot the full pre-compaction context, (b) measure what the summary lost, or (c) re-ground the agent from an external evidence store rather than the harness's own lossy summary. Identify prior art for 'compaction as a first-class observable event' and quantified context-loss metrics. Cite primary sources."

- **[DR]** "Research daemon-death forensics and heartbeat-liveness patterns for long-running single-writer local services (systemd Restart policies, faulthandler thread-stack dumps on SIGTERM, atexit clean-vs-vanish sentinels, heartbeat-age vs pid-file liveness). What is the state of the art for a service proving 'I am alive and my last work committed' vs merely 'a process exists', and for reconciling half-finished derived-state work after an unclean crash? Include concrete Python patterns and the failure modes of pid-file-based liveness."

---

## [acd34319709979f37] Performance & interactive latency — 15 ideas

Grounded in the query builders, the `actions` VIEW, FTS/vec0 runtimes, indexes, and connection profiles. Here is my lane.

## Performance & interactive latency — 15 ideas

- **vec0 semantic search is exact brute-force, not ANN** — `WHERE embedding MATCH ? AND k = ?` (`sqlite_vec_queries.py:129`) linearly scans *every* stored 1024-dim float32 vector (~4KB each × millions = multi-GB read per query). This is the single biggest interactive-tier latency at 38GB; sqlite-vec ships no IVF/HNSW index. Needs a coarse partition/prefilter or clustering strategy. — NEW
- **Binary/int8 quantize the vec0 store + float rerank** — cut the linear-scan bytes ~32× with a binary first pass over `message_embeddings`, then rerank top-K against float vectors. Directly attacks the brute-force scan above without changing recall much. — NEW (DR candidate)
- **Session-seeded similarity fans out N full-scan KNN queries** — `query_by_session` (`sqlite_vec_queries.py:175`) loops `_SESSION_SEED_FANOUT` seed vectors, each its own exact KNN scan; `find_similar_sessions`/neighbor tools pay N× the linear scan. Batch into one multi-probe pass or precompute session centroids. — NEW
- **`actions` VIEW is a blocks self-join on every action/tool query** — `LEFT JOIN blocks r ON r.tool_id=u.tool_id AND r.session_id=u.session_id AND r.block_type='tool_result'` (`index.py:324`) re-joins the largest table each read; `tool_usage.py:160` does `LEFT JOIN actions a` *per session* in an aggregate. Verify `idx_blocks_type_tool`/`idx_blocks_tool_id` actually serve the join, and consider materializing `actions` as an index-tier table (it's rebuildable) for hot analyze paths. — NEW
- **FTS session-search window-sorts the entire candidate set before LIMIT** — `ROW_NUMBER() OVER (PARTITION BY session_id ...)` in `query_builders.py:74` materializes every block hit for a term before taking top-N; a high-frequency word ("error", "test") matches millions of rows → worst-case full sort with no early cutoff. Cap candidate hits (per-session dedup via a bounded subquery) before the window. — NEW
- **Preview `COUNT(*)` fires on every keystroke** — `sessions_reads.py:383` (`SELECT COUNT(*) FROM sessions {where}`) and message pagination `COUNT(*)` (`message_query_reads.py:326,418`) are exact O(rows) scans just to show a total. Serve capped/approximate counts ("500+") or defer the count off the interactive path. — NEW
- **Session listing uses OFFSET pagination** — `sessions_reads.py:216,312` append `LIMIT ? OFFSET ?`, which scans+discards; deep paging degrades linearly. Messages already use keyset (`message_query_reads.py:464`, good) — convert session listing to keyset on `(sort_key_ms DESC, session_id DESC)` matching `idx_sessions_origin_sort`. — NEW
- **The sort_key is a per-row COALESCE expression, unindexed** — `COALESCE(m.occurred_at_ms, s.sort_key_ms, s.updated_at_ms, s.created_at_ms, 0)` is both the ORDER BY and the `since` predicate in every ranked search (`query_builders.py:52,67`); the sort can't use an index, forcing a full sort of the candidate set. Persist a single resolved `effective_sort_ms` column on messages/blocks and index it. — NEW
- **Text-PK join chain FTS→blocks→messages→sessions on every hit** — `query_builders.py:56-58` joins `messages` and `sessions` by TEXT primary keys per hit; at high match counts these string-key probes dominate. Denormalize `origin` + `effective_sort_ms` onto `blocks` so ranking/scope filter needs zero session/message join. — NEW
- **FTS readiness probe runs before every retrieval** — `message_fts_search_readiness_async` is called on each `search_*` (`sessions_search.py:23,54,109`) as a hard correctness gate; if that probe is a COUNT or multi-table scan it adds fixed latency to every keystroke. Confirm it's O(1) (metadata/`PRAGMA`), not a row count. — NEW
- **Read connections under-provisioned for a 38GB archive** — `READ_CACHE_SIZE_KIB=32MiB`, `READ_MMAP_SIZE_BYTES=128MiB` (`connection_profile.py:84,87`) vs a large index.db; interactive readers page-fault on hot indexes. mmap is nearly free (page-cache mapping) — raise read mmap toward the index.db size and profile hot-index residency. — NEW
- **No EXPLAIN QUERY PLAN regression lane** — nothing captures golden query plans, so an index/schema change can silently flip a hot query (ranked search, actions aggregate, session list) to `SCAN`. Add a `devtools lab perf query-plans` snapshot lane asserting SEARCH-via-index on the interactive query set. — NEW
- **Transcript composition read-amplification on deep forks** — iterative composition (recent `1e4a69438`) replays the parent prefix on each read; at 38GB a hot deep-fork session re-composes per request. Measure the amplification and cache a composed-transcript materialization for frequently-read leaves. — NEW / relates to lineage #2467, #2470
- **No result/prefix cache for incremental preview** — typing streams `d→da→dat→data`, each a full FTS+window+join. A small debounce + LRU keyed on `(normalized_query, limit, scope)` (or prefix-result reuse) eliminates redundant scans on the exact interactive path that matters most. — NEW
- **Interactive analyze verbs scan full tables** — `stats.py`/`get_stats_by`/`cost_rollups` aggregate over messages/blocks with no pre-aggregation; `analyze`/facets can hit multi-second scans on the interactive tier. Add materialized rollup tables (index tier, rebuildable) for the common group-by facets. — NEW

---

## GPT-pro prompt stubs

**[DR] SQLite vector search at scale** — "In SQLite using the `sqlite-vec` `vec0` virtual table, semantic search over ~several million 1024-dim float32 embeddings (~10+ GB) is currently an exact brute-force `WHERE embedding MATCH ? AND k=?` linear scan, and there's no session/origin prefilter. Research the current (2025–2026) state of: (a) approximate-nearest-neighbor options that coexist with `vec0` in a single-writer SQLite file (IVF/partitioned metadata columns, external HNSW sidecar, DiskANN-style), (b) binary/int8 quantization + float rerank tradeoffs on recall vs scan-byte reduction, and (c) multi-probe batching to replace per-seed KNN fan-out. Give concrete latency/recall numbers and a migration path that keeps SQLite as the sole store."

**[A] `actions` VIEW vs materialized table** — "A read-heavy analytics system derives an `actions` relation as a SQL VIEW that LEFT-JOINs a `blocks` table to itself (`tool_use` ↔ `tool_result` by `tool_id`+`session_id`). It's the largest table and the view is queried on hot per-session aggregate paths. Analyze the join-cost profile and lay out the decision criteria and mechanics for materializing it as an index-tier (rebuildable) table with the right covering indexes, including staleness/rebuild handling."

**[A] Sub-second interactive count + pagination** — "Design the query strategy for an interactive session browser over ~16k sessions / 4.3M messages in SQLite where (1) a total-result count currently fires exact `COUNT(*)` on every keystroke and (2) session listing uses `LIMIT/OFFSET`. Specify approximate/capped counting, keyset pagination on `(sort_key DESC, id DESC)`, and how to keep the two consistent, with the index shapes required."