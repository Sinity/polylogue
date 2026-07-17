# Polylogue R&D bundle — file 4 of 6

> Preamble below is the MASTER SYNTHESIS, identical across all 6 files.
> The unique payload of THIS file is the 16 subagent outputs after the PAYLOAD divider.
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
  PART 2 - PAYLOAD: 16 subagent outputs (unique to file 4 of 6)
=====================================================================


---

## [a1c3322c559372e7e] WAVE-4 SPEC — Ingest Fidelity & Coverage

I have the grounding I need. Delivering the spec.

---

# WAVE-4 SPEC — Ingest Fidelity & Coverage

**"Did we lose anything on ingest, and can we prove it?"**
Grounded in `sources/dispatch.py` (902 L), `raw_sessions` DDL (`storage/sqlite/archive_tiers/source.py`, `SOURCE_SCHEMA_VERSION=2`), and `core/enums.py` (`ValidationStatus`, `ValidationMode`, `ArtifactSupportStatus`, `Origin`).

## 0. Framing invariant

Fidelity is the gap between **acquired bytes** (`raw_sessions.blob_hash`/`blob_size`, durably stored, `blob_refs.ref_type='raw_payload'`) and the **parsed tree** (`sessions→messages→blocks` in rebuildable `index.db`). Because the daemon is the sole writer and re-ingest is content-hash idempotent, fidelity is a *pure function of (blob, parser)* — so almost everything here is **derived/rebuildable**, with one durable exception (`parser_fingerprint`) that must survive an index rebuild to avoid a full re-parse.

---

## 1. Schema additions (+ tier assignment)

### 1a. DURABLE — `source.db` migration `003_parser_fingerprint.sql` (bumps `SOURCE_SCHEMA_VERSION` 2→3)

Additive-only, one `PRAGMA user_version` step, behind a verified backup manifest (durable-tier regime). Two nullable columns; STRICT-safe.

```sql
-- raw_sessions: the parse-identity that forces reprocess-on-improvement
ALTER TABLE raw_sessions ADD COLUMN parser_fingerprint TEXT;   -- NULL = never parsed under fingerprint regime

-- raw_artifacts: structured decode-failure taxonomy alongside existing free-text decode_error
ALTER TABLE raw_artifacts ADD COLUMN decode_failure_class TEXT
    CHECK (decode_failure_class IS NULL OR decode_failure_class IN
        ('encoding','framing','schema','truncation','empty','none'));
```

- **`parser_fingerprint`** lives next to the existing durable parse-outcome fields (`parsed_at_ms`, `parse_error`) — precedent: parse outcome already persists in the durable source tier because it gates `idx_raw_sessions_parse_ready`. Value = `fingerprint_hash((detector_id, parser_id, parser_semver, dispatch_rules_hash))` reusing the existing `schemas/observation_identity.py:fingerprint_hash` convention (short stable hash, no new hashing primitive).
- **`decode_failure_class`** promotes the untyped `raw_artifacts.decode_error` into a closed taxonomy. New enum `DecodeFailureClass` in `core/enums.py` (mirrors the `ArtifactSupportStatus` pattern): `encoding` (bytes→text failed — bad charset/BOM/UTF-8), `framing` (JSONL line/record boundary broke — feeds `malformed_jsonl_lines`), `schema` (decoded fine, no detector claimed / Pydantic reject), `truncation` (tail cut mid-record — see `_trim_jsonl_detection_prefix`), `empty` (0 bytes / whitespace), `none`.

### 1b. REBUILDABLE — `index.db` new table `raw_fidelity` (edit canonical DDL, bump `INDEX_SCHEMA_VERSION` 24→25, **no migration chain** — rebuild via `polylogue ops reset --index && polylogued run`)

```sql
CREATE TABLE IF NOT EXISTS raw_fidelity (
    raw_id                  TEXT PRIMARY KEY,          -- 1:1 with raw_sessions.raw_id
    origin                  TEXT NOT NULL CHECK (...Origin...),
    blob_size               INTEGER NOT NULL,          -- denormalized from raw_sessions for read-locality
    parsed_char_bytes       INTEGER NOT NULL,          -- Σ len(block.text.encode) + structural bytes attributed
    byte_fidelity_ratio     REAL NOT NULL,             -- parsed_char_bytes / NULLIF(blob_size,0), clamped [0,1+ε]
    roundtrip_status        TEXT NOT NULL CHECK (roundtrip_status IN
                              ('exact','structural','lossy','skipped','failed')),
    roundtrip_hash_match    INTEGER NOT NULL DEFAULT 0 CHECK (roundtrip_hash_match IN (0,1)),
    unparsed_key_count      INTEGER NOT NULL DEFAULT 0,
    unparsed_key_census_json TEXT NOT NULL DEFAULT '{}', -- {dotted.key.path: {count, sample_type}}
    detection_ambiguity_score REAL NOT NULL DEFAULT 0.0, -- 0=unique claim, →1 = many detectors matched
    detection_runner_up     TEXT,                        -- 2nd-place detector id, NULL if unique
    misclassification_flag  INTEGER NOT NULL DEFAULT 0 CHECK (misclassification_flag IN (0,1)),
    zero_message_anomaly    INTEGER NOT NULL DEFAULT 0 CHECK (zero_message_anomaly IN (0,1)),
    drop_accounting_json    TEXT NOT NULL DEFAULT '{}',   -- {stream_window_blind, truncated_tail, depth_cap, malformed_lines}
    computed_at_ms          INTEGER NOT NULL,
    parser_fingerprint      TEXT                          -- copy at compute time; staleness = != raw_sessions value
) STRICT;
CREATE INDEX IF NOT EXISTS idx_raw_fidelity_low ON raw_fidelity(byte_fidelity_ratio) WHERE byte_fidelity_ratio < 0.90;
CREATE INDEX IF NOT EXISTS idx_raw_fidelity_flags ON raw_fidelity(origin)
  WHERE misclassification_flag = 1 OR zero_message_anomaly = 1 OR roundtrip_status = 'failed';
```

**Tier rationale:** `raw_fidelity` is 100% recomputable from `blob + parser`, so it is derived → `index.db`, no migration, rebuilt during the **materialize** stage. Only `parser_fingerprint` (the *decision input*, not the *metric*) is durable. This split is the load-bearing architectural call: never put a derivable ratio in a durable tier that needs a backup manifest to evolve.

### 1c. Detection-ambiguity requires a dispatch API addition (not a schema change)

`detect_provider()` today returns first-match-wins in tightness order (`sources/dispatch.py:178`). Add a sibling **`detect_all(payload, path) -> list[DetectionCandidate]`** returning every detector that would claim the payload, ordered by tightness, each with a confidence band (`structural` > `pydantic-validated` > `loose-dict-key`). `detect_provider` becomes `detect_all(...)[0]`. Ambiguity score is derived from this list (§2.3). This is the only production-path code change fidelity needs; everything else reads existing artifacts.

---

## 2. Algorithms (pseudocode)

### 2.1 Byte-fidelity ratio + round-trip reconstruction

```
def compute_fidelity(raw_id):
    blob      = source.read_blob(raw_id)            # via blob_refs ref_type='raw_payload'
    blob_hash = raw_sessions[raw_id].blob_hash      # 32-byte, trusted
    parsed    = index.load_tree(raw_id)             # sessions→messages→blocks derived from this raw_id

    # (a) byte-fidelity ratio: how much acquired content survived into blocks
    parsed_bytes = sum(len(b.search_text.encode('utf-8')) for b in parsed.blocks) \
                 + attributed_structural_bytes(parsed)   # titles, tool args/results, ts
    ratio = parsed_bytes / max(blob_size, 1)
    # ratio is a *coverage proxy*, NOT equality — JSON scaffolding inflates blob_size,
    # so define per-origin expected-ratio bands (JSONL runtime ≈ 0.35–0.70,
    # single-doc export ≈ 0.55–0.85); flag deviation from the band, not absolute < 1.

    # (b) round-trip: re-serialize parsed tree back to provider-native shape, re-hash
    if parser_supports_reserialize(origin):
        rebuilt      = reserialize_to_native(parsed, origin)   # inverse of the parser
        rebuilt_hash = sha256(nfc_normalize(rebuilt))          # SAME normalization as pipeline/ids.py
        if rebuilt_hash == blob_hash:            roundtrip='exact';      match=1
        elif canonical_json(rebuilt)==canonical_json(json(blob)): roundtrip='structural'; match=0
        else:                                    roundtrip='lossy';      match=0
    else:
        roundtrip='skipped'; match=0             # grok-export, browser-capture, etc.
    return ratio, roundtrip, match
```

Round-trip is **tiered on purpose**: byte-exact is unattainable for most providers (key reordering, whitespace, our own NFC pass), so `structural` (canonical-JSON equality after normalization) is the real pass bar; `lossy` means keys/values were dropped and is the tripwire. `exact` is reserved for providers with a byte-stable serializer.

### 2.2 Unparsed-key census

```
def unparsed_key_census(raw_id):
    doc          = json_or_jsonl(blob)                 # tolerant: skip malformed lines, count them
    consumed     = parser_consumed_keypaths(raw_id)    # instrument parser to emit every key it read
    census = defaultdict(lambda: {"count":0, "sample_type":None})
    for keypath, value in walk_all_keypaths(doc):      # dotted paths, arrays as [*]
        if keypath not in consumed:
            census[keypath]["count"]  += 1
            census[keypath]["sample_type"] = json_type(value)
    # drop known-ignorable noise (pretty-print keys, provider echo fields) via per-origin allowlist
    census = {k:v for k,v in census.items() if k not in IGNORABLE_KEYS[origin]}
    return len(census), census
```

`parser_consumed_keypaths` is obtained by wrapping the payload dict in a **read-tracking proxy** (records every `__getitem__`/`.get`) for one parse pass — no parser rewrite, just an instrumentation harness invoked at materialize time. The census is the primary signal for "a provider added a field we silently drop."

### 2.3 Detection-ambiguity score + post-parse misclassification tripwire

```
def detection_ambiguity(payload, path):
    cands = detect_all(payload, path)          # §1c: every detector that would claim it
    if len(cands) <= 1: return 0.0, None
    top, runner = cands[0], cands[1]
    # tightness gap: structural-vs-loose = low ambiguity; two loose-dict claims = high
    gap = tightness_rank(runner.band) - tightness_rank(top.band)
    score = 1.0 / (1.0 + gap)                   # same band → 0.5+, big gap → →0
    return score, runner.detector_id

def misclassification_tripwire(raw_id):
    # post-parse: does the CHOSEN origin's output look like the origin?
    flags = []
    if parsed.message_count == 0 and blob_size > EMPTY_FLOOR: flags.append('zero_msg_nonempty')
    if origin_expected_roles(origin) not disjoint parsed.roles is False: flags.append('role_shape_mismatch')
    if ratio far outside per-origin band:                    flags.append('ratio_out_of_band')
    if origin==UNKNOWN_EXPORT and detect_all found a real candidate: flags.append('funneled_but_claimable')
    return 1 if flags else 0, flags
```

The tripwire is **post-parse confirmation**: detection is a *prediction*; after parsing we check the parsed shape matches the claimed origin's expected shape (role vocabulary, message-count distribution, ratio band). A `role_shape_mismatch` on a confident detection is the highest-value misclassification signal.

### 2.4 Streaming blind-spot + truncated-tail + depth-cap drop accounting

Three loss channels intrinsic to `dispatch.py`, each recorded in `drop_accounting_json`:

- **stream-window blind spot** — `_detect_provider_from_raw_bytes` samples only `islice(stream, 32)` (line 212) for JSONL detection. A multi-GB Codex/Claude Code file whose provider signature only appears after record 32 detects via fallback. Accounting: `stream_window_blind = (total_records > 32 and detection_used_fallback)`.
- **truncated tail** — `_trim_jsonl_detection_prefix` (line 236) drops a partial final line when the file doesn't end in newline (live-appending hot file). Accounting: `truncated_tail_bytes = len(raw_bytes) - newline_at - 1` when trimmed; correlate with the hot-file quiet-deferral path.
- **depth-cap drops** — the memory-bounded streaming parser and any recursion cap in `_lower_payload_specs` nesting. Accounting: `depth_cap_drops = records_seen - records_lowered`.
- **malformed lines** — already counted in `raw_artifacts.malformed_jsonl_lines`; copied into `drop_accounting_json.malformed_lines` for one-stop read.

```
drop_accounting = {
  "stream_window_blind": bool, "records_total": N, "records_sampled": min(N,32),
  "truncated_tail_bytes": T, "depth_cap_drops": D, "malformed_lines": M,
}
```

### 2.5 Zero-message parse-success anomaly & unknown/grok funnel

```
zero_message_anomaly = (parse_error IS NULL           # parser claimed success
                        and validation_status != 'failed'
                        and parsed.message_count == 0
                        and blob_size > EMPTY_FLOOR)   # not a legitimately empty session

# unknown/grok funnel: sessions that landed on a no-parser origin
funnel = raw_sessions where origin in (UNKNOWN_EXPORT, GROK_EXPORT)
       joined to raw_artifacts.support_status  # recognized_unparsed vs unknown vs decode_failed
# grok-export is a reserved token with NO wired parser → every grok blob is 100% unparsed by design;
# surface it as a KNOWN coverage gap (support_status='recognized_unparsed'), NOT a fidelity failure.
```

The funnel view is the coverage backstop: it answers "which acquired bytes have *no path to a tree at all*" and separates *recognized-but-unparsed* (grok, expected) from *unknown* (detection genuinely failed — actionable) from *decode_failed* (bytes broke).

---

## 3. Reprocess-on-parser-improvement plan

The daemon already has `reprocess = parse+materialize+index` and a `convergence_debt` retry substrate. Wire `parser_fingerprint` as the trigger:

1. **Compute** the current fingerprint once at daemon start: `CURRENT_FP = fingerprint_hash((detector_rules_hash, parser_registry_semver))`. `detector_rules_hash` covers `dispatch.py` tightness order + each parser's version; bump the parser semver when a parser learns a new field (the census in §2.2 is what tells you to bump it).
2. **On parse**, stamp `raw_sessions.parser_fingerprint = CURRENT_FP`.
3. **Reprocess selection query** (new `ConvergenceStage: parser_drift`, `false_means_pending` so it drains in bounded windows):
   ```sql
   SELECT raw_id FROM raw_sessions
   WHERE parsed_at_ms IS NOT NULL
     AND (parser_fingerprint IS NULL OR parser_fingerprint != :CURRENT_FP)
   ORDER BY (parse_error IS NOT NULL) DESC,   -- previously-failed first: most likely to improve
            acquired_at_ms DESC
   LIMIT :window;
   ```
4. Each selected `raw_id` re-runs parse+materialize (no re-acquire — bytes are durable), re-stamps the fingerprint, and recomputes `raw_fidelity`. Idempotency holds: if the improved parser yields an identical tree, the content hash matches and the write is a no-op except the fingerprint bump.
5. **Guard rails:** bounded window per quiet cycle (don't re-parse 16K sessions in one tick); a `parser_fingerprint` bump is a deliberate, reviewed act (it's effectively a mass-reprocess switch) — gate it behind the same batching discipline the CLAUDE.md schema note demands ("batch same-tier bumps, don't repeatedly reset the active archive").

This is strictly better than an index reset for parser improvements: it re-parses *only* stale-fingerprint rows and keeps un-improved sessions untouched, whereas `ops reset --index` re-parses everything.

---

## 4. Test strategy

**Coverage-regression golden (the anti-silent-loss net):**
- A checked-in corpus of one representative blob per `Origin` (synthetic, private-data-free — use `SessionBuilder`/`polylogue demo seed`). For each, snapshot a **golden fidelity fingerprint**: `{byte_fidelity_ratio band, roundtrip_status, unparsed_key_count, sorted(census.keys())}`.
- Test asserts the census key-set is a **superset check that only shrinks**: if a parser change makes a previously-unparsed key parsed, the golden updates *down*; if a key regresses to unparsed, the test fails loud. This is the mechanism that catches "we stopped reading a field."
- Distinguish from a memorialize-a-diff test (which CLAUDE.md forbids): this guards the *fidelity contract* (no silent coverage loss), not a literal spelling.

**Misclassification tripwire test:**
- Adversarial cross-origin corpus: feed a ChatGPT blob and assert it is NOT claimed by the Codex/Claude Code detectors; feed a near-miss (ChatGPT-shaped with a Codex-ish key) and assert `detection_ambiguity_score > 0` with the correct `detection_runner_up`.
- Assert the post-parse tripwire fires `role_shape_mismatch` when a blob is force-parsed under the wrong origin (inject via `_parse_lowered_spec` with a wrong `Provider`).

**Round-trip property test (Hypothesis, `tests/property`):**
- For providers with a reserializer: `SessionBuilder`-generated tree → serialize-to-native → re-parse → assert tree equality (structural round-trip is total). This is the invariant, not the ratio.

**Streaming blind-spot regression:**
- Synthesize a JSONL where the provider signature first appears at record 40; assert `drop_accounting.stream_window_blind == True` and that detection still *eventually* resolves via the record-shape path (not silently fallback-to-unknown). Guards the `islice(…, 32)` boundary.

**Zero-message anomaly:**
- Blob with valid frames but no message-bearing records → assert `zero_message_anomaly=1` and it surfaces in the funnel view, not as a clean parse.

Run through `devtools test <file>` (testmon inner loop) — fidelity tests touch `dispatch.py` + source/index DDL so testmon selects them on any parser or schema change. Clock-sensitive computes use `frozen_clock`.

---

## 5. Bead breakdown (acceptance criteria)

| # | Bead | Scope | Acceptance |
|---|------|-------|-----------|
| **F1** | `feat(dispatch): add detect_all + DetectionCandidate` | §1c — `detect_all()` returns ranked candidates w/ confidence band; `detect_provider` delegates to `[0]`. No behavior change to existing detection. | `detect_provider` output byte-identical on the golden corpus; `detect_all` returns ≥1 candidate for every non-empty blob; unit test proves ordering = tightness order. |
| **F2** | `feat(schema): parser_fingerprint + decode_failure_class (source 003)` | §1a — additive durable migration, `SOURCE_SCHEMA_VERSION` 2→3, `DecodeFailureClass` enum, backup manifest. | `devtools lab policy schema-versioning` passes (additive, single `user_version` step); existing archives migrate with no data loss; `nullable_check` CHECK renders; write path stamps fingerprint. |
| **F3** | `feat(index): raw_fidelity table + materialize compute` | §1b + §2.1/2.2 — DDL, `INDEX_SCHEMA_VERSION` 24→25, byte-ratio + round-trip + census computed in materialize stage; read-tracking proxy harness. | Index rebuild populates `raw_fidelity` 1:1 with `raw_sessions`; ratio ∈ [0, 1+ε]; census keys ⊆ blob keys; `render all --check` clean (topology projection regenerated). |
| **F4** | `feat(fidelity): ambiguity score + misclassification tripwire` | §2.3 — score from `detect_all`, post-parse role/ratio/zero-msg checks, flags into `raw_fidelity`. | Adversarial cross-origin corpus: no false misclassification on clean golden; tripwire fires on injected wrong-origin parse; ambiguity=0 for uniquely-claimed blobs. |
| **F5** | `feat(fidelity): drop-accounting + zero-msg + unknown/grok funnel` | §2.4/2.5 — populate `drop_accounting_json`, `zero_message_anomaly`; funnel view separating recognized_unparsed / unknown / decode_failed. | `stream_window_blind` true iff records>32 & fallback used; grok blobs classify `recognized_unparsed` (not failure); funnel view sums to all `UNKNOWN_EXPORT`+`GROK_EXPORT` rows. |
| **F6** | `feat(daemon): parser_drift reprocess ConvergenceStage` | §3 — fingerprint-mismatch selection, bounded-window `false_means_pending`, re-stamp + recompute fidelity. | Bumping a parser semver enqueues only stale-fingerprint rows; identical-tree re-parse is a content-hash no-op; window bound respected; no full-archive re-parse in one tick. |
| **F7** | `test(fidelity): coverage-golden + tripwire + roundtrip suites` | §4 — golden snapshots, misclassification, Hypothesis round-trip, streaming/zero-msg regressions. | Golden fails on a synthetic dropped-key regression; round-trip property total for reserializable origins; testmon selects the suite on `dispatch.py`/DDL edits. |
| **F8** *(opt)* | `feat(surface): fidelity read surface (CLI/MCP)` | Expose `raw_fidelity` via a query-first read / MCP tool (`archive_debt`-adjacent) — "show low-fidelity / misclassified / unparsed-key sessions". | New MCP tool → `EXPECTED_TOOL_NAMES` + contract updated; CLI verb renders ratio/flags; discovery tests pass. |

Dependency order: **F1 → F2 → F3 → {F4, F5} → F6**; **F7** rides alongside F3–F6; **F8** last. F2 (durable migration) is the serialization point — batch it as the single durable-tier bump for this wave.

---

## 6. Top-3 risks

1. **Byte-fidelity ratio is a coverage proxy, not equality — easily misread as a bug.** Blob bytes include JSON scaffolding, provider echo fields, and base64 attachment refs that legitimately never enter blocks, so a healthy session sits well below 1.0. If the ratio is surfaced as a raw "% preserved," every session looks broken. **Mitigation:** per-origin expected-ratio *bands* (calibrated from the golden corpus), flag *deviation from band* not absolute value; document that round-trip `structural` equality — not the ratio — is the real loss test.

2. **Round-trip requires a reserializer polylogue doesn't have for every origin.** The parser is lossy-by-design for several providers (browser-capture, single-doc exports, and grok has no parser at all). A naive "round-trip must match blob_hash" gate would fail-closed on the majority of the archive. **Mitigation:** the `skipped`/`structural`/`lossy` tiering (§2.1) makes round-trip *opt-in per origin*; only `lossy` is a tripwire. Never block ingest on round-trip — it's a materialize-time diagnostic, never a parse gate.

3. **`parser_fingerprint` is a mass-reprocess switch that can thrash the daemon.** A fingerprint bump marks *every* prior parse stale; an unbounded reprocess against a 38GB / 16K-session archive re-parses everything and stalls convergence (the exact #1498-cascade failure mode the memory warns about). **Mitigation:** F6's bounded-window `false_means_pending` stage + treating any fingerprint bump as a reviewed, batched act (CLAUDE.md schema-batching discipline); the fingerprint composition must be *stable* across daemon restarts (hash of rules+semver, not a timestamp) or it self-triggers a perpetual full re-parse.

---

**Grounding cites:** `raw_sessions`/`blob_refs`/`raw_artifacts` DDL at `storage/sqlite/archive_tiers/source.py:15-108`; enums at `core/enums.py:42,332,346,360`; detection + `islice(…,32)` window at `sources/dispatch.py:178,212`; truncated-tail trim at `dispatch.py:236`; reusable hash at `schemas/observation_identity.py:29`; existing `require_unparsed` selection at `pipeline/services/planning_backlog.py:32`. No code or beads modified.

---

## [a1508a04e9e998413] SPEC — Delegation unit + delegation-card + analytics DSL + `delegation` polydemo

I have complete grounding. Here is the implementation-grade spec.

---

# SPEC — Delegation unit + delegation-card + analytics DSL + `delegation` polydemo

Scope owner: the delegation *product algebra* (unit → card → DSL → yield measure → demo). Grounded in live `index.db` v24: `session_links(link_type='subagent')`, the `actions` VIEW, `session_profiles`, `session_runs(role='subagent')`, `messages.model_name`/`material_origin`, and the existing demo harness (`polylogue/demo/{constructs,tour,seed,verify}.py`). Bead 212 ground rule inherited: **every displayed number resolves on `--explain` to structural evidence, never regex over prose.**

## 0. What a "delegation" IS (and is not)

A **delegation** is a *parent-dispatched* subagent edge: a `session_links` row with `link_type='subagent'` **that also has a parent-side Task dispatch action**. The dispatch action is the `actions`-VIEW row (a `tool_use` block, `tool_name='Task'`/`tool='subagent'`) whose message is at/near `branch_point_message_id`; its paired `tool_result` block is the returned artifact and carries `is_error`/`exit_code`.

Not every `subagent` link is a delegation, and this is the load-bearing construct-validity gate:
- Codex async subagents / sidechains with no parent Task action → `result_status='unknown'`, excluded from ROI denominator.
- `inheritance='prefix-sharing'` continuations/auto-compaction are *not* delegations (different `link_type`).

So the delegation relation = `session_links(subagent)` **LEFT JOIN** the parent Task action, and we keep the unmatched ones visible but honestly labeled `unknown`.

---

## 1. Schema / DDL — `delegations` VIEW + one enabling derived column

**Table vs VIEW decision: VIEW.** Delegations are 100% derivable from `index.db` (rebuildable tier), and the precedent is exact — `actions` is a VIEW, not a table. A materialized table would add a convergence stage for no durability gain. The only genuinely per-row-expensive field (dominant model of a session) is precomputed into `session_profiles` as an additive derived column, so the VIEW stays cheap joins.

### 1a. Enabling primitive — additive derived columns on `session_profiles` (index tier)

The missing primitive (per the compositionality rule: "if a demo needs bespoke logic, file the primitive first"). Model/family is not first-class today — it lives in `messages.model_name` and `per_model_cost_json`. Add:

```sql
-- session_profiles (index.db, derived tier — DDL edit + rebuild, NO migration chain)
primary_model_name    TEXT,           -- dominant model_name by assistant output-token share
primary_model_family  TEXT,           -- canonical family via core.sources mapping (NOT wire token)
```

Populated at profile materialization from the session's `messages` where `material_origin='assistant_authored'`, dominant = max Σ`output_tokens` per `model_name`. `primary_model_family` maps through a **canonical model→family function in `core/sources.py`** (e.g. `fable`/`opus`/`sonnet`/`haiku` → `anthropic`; gpt-5.x/codex → `openai`; deepseek → `deepseek`). This mapping is the origin-vocab projection boundary (see Risk 1).

### 1b. `delegations` VIEW

```sql
CREATE VIEW IF NOT EXISTS delegations AS
SELECT
    l.src_session_id                       AS parent_session_id,
    l.resolved_dst_session_id              AS child_session_id,
    l.branch_point_message_id              AS dispatch_message_id,
    l.confidence                           AS link_confidence,
    l.method                               AS link_method,
    l.inheritance                          AS inheritance,
    -- orchestrator (parent) identity
    pp.primary_model_name                  AS orchestrator_model,
    pp.primary_model_family                AS orchestrator_model_family,
    p.origin                               AS orchestrator_origin,
    p.repo_id                              AS repo_id,        -- for session.repo pushdown
    pp.terminal_state                      AS parent_terminal_state,
    -- subagent (child) identity + spend
    cp.primary_model_name                  AS subagent_model,
    cp.primary_model_family                AS subagent_model_family,
    cp.total_cost_usd                      AS child_cost_usd,
    cp.cost_is_estimated                   AS child_cost_is_estimated,
    (cp.total_input_tokens + cp.total_output_tokens
       + cp.total_cache_read_tokens + cp.total_cache_write_tokens) AS child_tokens,
    cp.wall_duration_ms                    AS child_wall_ms,
    cp.terminal_state                      AS child_terminal_state,
    -- dispatch action = instruction payload + returned artifact + provider outcome
    a.tool_use_block_id                    AS instruction_block_id,
    a.tool_input                           AS instruction_payload,
    a.tool_result_block_id                 AS artifact_block_id,
    a.output_text                          AS artifact_text,
    a.is_error                             AS result_is_error,   -- NULL = unknown (never guessed)
    a.exit_code                            AS result_exit_code,
    CASE
        WHEN a.tool_use_block_id IS NULL      THEN 'unknown'  -- no parent Task action
        WHEN a.is_error IS NULL               THEN 'unknown'  -- provider didn't report
        WHEN a.is_error = 1                   THEN 'error'
        ELSE 'ok'
    END                                    AS result_status
FROM session_links l
JOIN sessions p             ON p.session_id  = l.src_session_id
LEFT JOIN sessions c        ON c.session_id  = l.resolved_dst_session_id
LEFT JOIN session_profiles pp ON pp.session_id = l.src_session_id
LEFT JOIN session_profiles cp ON cp.session_id = l.resolved_dst_session_id
LEFT JOIN actions a
       ON a.session_id = l.src_session_id
      AND a.message_id = l.branch_point_message_id
      AND a.semantic_type = 'subagent'      -- the Task/subagent dispatch, not sibling tool calls
WHERE l.link_type = 'subagent'
  AND (l.status IS NULL OR l.status <> 'quarantined');
```

Every column resolves to a structural source; `result_status` derives *only* from `actions.is_error`/`exit_code` (which come from `blocks.tool_result_is_error`, the index-v16 keystone). `index.db` schema bump **24 → 25**.

---

## 2. Extraction algorithm (already 90% built — this is composition, not new parsing)

The spawn edge + `branch_point_message_id` are already produced by the existing lineage parsers (`session_links` upsert + `resolve_session_links_for_session`). Delegation extraction is **materialization-time composition**, not a new source parser:

```
# runs inside session_profiles materialization (per session S):
def derive_primary_model(S):
    tallies = {}                       # model_name -> output_tokens
    for m in messages(S) where material_origin == 'assistant_authored' and model_name is not None:
        tallies[m.model_name] += m.output_tokens
    if not tallies: return (None, None)
    primary = argmax(tallies)          # dominant by output-token share
    return (primary, canonical_family(primary))   # core.sources mapping; origin-vocab

# delegations require NO extra step — the VIEW composes at read time:
#   parent Task dispatch  = actions row at branch_point_message_id, semantic_type='subagent'
#     - instruction payload = tool_use.tool_input
#     - artifact            = paired tool_result.text
#     - result_status       = tool_result.is_error / exit_code  (NULL -> 'unknown')
#   child spend            = child session_profiles (cost/tokens/wall) — already materialized
#   models                 = primary_model_family on both profiles
```

Provider specifics (structural anchors only, no prose mining):
- **Claude Code:** `Task` tool_use → sidechain child (`branch_type='subagent'`, `link_type='subagent'`); `tool_result` is the subagent's returned summary → `result_status` reads directly. Already exercised by demo constructs `subagent_links`, `subagent_run_rows`, `subagent_context_snapshots (boundary='subagent_start')`.
- **Codex:** async subagent threads link as `subagent` but frequently lack a parent Task action row → `result_status='unknown'`. Correct and honest: counted, ROI-excluded.

---

## 3. Analytics DSL — `delegation` unit

Register one `QueryUnitDescriptor` + `StructuralQueryUnitInfo` (pattern from `metadata.py:740-868`), backed by a `query_delegations` `sql_query_method` over the VIEW.

```python
QueryUnitDescriptor(
    "delegation", "delegation", "delegations",
    exists_supported=True,
    payload_model="DelegationQueryRowPayload",
    sql_query_method="query_delegations",
    cli_plain_renderer="delegation",
    aggregate_group_fields=(
        "subagent_model_family", "orchestrator_model_family",
        "result_status", "parent_terminal_state",
        "session.origin", "session.repo",
    ),
    fields=_unit_info("delegation").fields,
    terminal_example=
      "delegations where orchestrator_model:fable | group by subagent_model_family | count",
)
```

Fielded predicates: `orchestrator_model`, `orchestrator_model_family`, `subagent_model`, `subagent_model_family`, `result_status`(ok|error|unknown), `exit_code`, `child_cost_usd` (range), `child_tokens` (range), `child_wall_ms` (range), `parent_terminal_state`, `child_terminal_state`, `inheritance`, `link_confidence` (range), `session.repo`, `session.origin`. `session.repo`/`session.origin` push down through `repo_id`/`orchestrator_origin`; the rest are VIEW-column predicates. Pipeline `where … | group by … | count` reuses the generic aggregate lowerer (no new machinery). `explain_query_expression` renders the lowered SQL for the demo's "prove it means what it says" beat.

### Delegation-yield measure

A `group by` projection (registered as a transform/aggregate, not bespoke python) computing per group:

- `delegations` — count
- `child_cost_total` — Σ`child_cost_usd` (with `cost_is_estimated` coverage footnote, mirroring D2 exact-vs-catalog honesty)
- `ok_cost` — Σ`child_cost_usd` where `result_status='ok'` **AND** `parent_terminal_state='clean_finish'`
- `wasted_cost` — Σ where `result_status='error'` OR `parent_terminal_state IN ('error_left','question_left','tool_left')`
- `unknown_cost` — Σ where `result_status='unknown'` (the honest "cannot attribute" bucket, never folded into either)
- **`yield_ratio = ok_cost / (ok_cost + wasted_cost)`** — ROI, computed only over the *attributable* denominator; `unknown_cost` reported alongside, never hidden.
- `mean_child_wall_ms_ok`, `mean_child_tokens_ok`

---

## 4. `delegation-card` render layout

A **read-view render profile** registered in the read-view profile registry (surfaces via `list_read_view_profiles`) — a product primitive, not a script. One card per delegation row; `--explain` appends the resolving block/profile ref to each line.

```
DELEGATION   parent ⇒ subagent
  orchestrator  Fable (anthropic)         parent  claude-code:…a1  repo polylogue  ⟶ clean_finish
  subagent      Sonnet (anthropic)        child   claude-code:…b7  role subagent
  instruction   "Audit the delegations VIEW for construct-validity gaps…"   [blk …:12 use]
  artifact      "3 gaps found: unknown-status not excluded from ROI; …"      [blk …:19 result]  exit 0 · error no
  cost          child $0.42 (exact) · 18.3k tok · 44.1s wall
  outcome       result ok · parent clean_finish
  yield         attributable ok-cost 0.42 / 0.42 = 100%     (unknown-cost 0.00 excluded)
```

`unknown` result renders `outcome  result unknown (no provider outcome reported)` and the yield line reads `yield  unattributable — excluded from ROI`. Plain, JSON, and MCP outputs all project through the same `INSIGHT_REGISTRY`-style descriptor so the three surfaces stay in lockstep; model tokens project origin-vocab at the boundary.

---

## 5. `delegation.polydemo` — executable demo

Follows the `tour.py` template (real `polylogue` CLI subprocess steps against the deterministic seed-1843 corpus; narration in `explanation`; construct assertions in `constructs.py`). Per the compositionality rule, **every step is a product primitive** (DSL query, delegation-card read, `explain_query_expression`) with shell/python only as sequencing glue — matching the `agent_forensics.py → polylogue analyze` fold.

**Comedic hook (Fable "iron fist"):** seed a Fable orchestrator session that fan-outs to a squad of subagents of mixed families/outcomes. Narration frames Fable as a micromanaging tyrant ruling its subagents with an iron fist — *then the receipts land*.

**Rigorous finding (deterministic on seed 1843):** the ROI query shows the iron fist is not free — e.g. Fable→`anthropic` subagents return `yield_ratio` X% while a chunk of spend sits in `error`/abandoned or `unknown` buckets; the single most expensive delegation is drillable to its exact `tool_result` block. The punchline is a *number that resolves*, not a joke.

Step sequence (each a real CLI call, exit 0, within `tour.py` budgets):
1. `analyze --facets` — archive overview (sets the stage).
2. `find "delegations where orchestrator_model:fable" then read --view delegation-card --limit 3` — the iron-fist roster.
3. `explain_query_expression "delegations where orchestrator_model:fable | group by subagent_model_family | count"` — prove the query means what it says (bead 212.4 beat).
4. `find "delegations where orchestrator_model:fable | group by subagent_model_family | count"` — the yield/ROI table.
5. `find "delegations where result_status:error" then read --view delegation-card --limit 1 --explain` — the most expensive failure, drilled to the raw `tool_result` block.
6. One honest "cannot answer" line: the `unknown`-status count (Codex async subagents with no provider outcome) — construct-validity slide.

Seed additions (in `seed.py`): a Fable-orchestrated Claude Code session with ≥3 Task dispatches to ≥2 subagent families, ≥1 `result_status='error'`, ≥1 `unknown` (Codex-origin subagent, no Task action), deterministic costs so yield numbers are fixed.

New `DemoConstruct`s (in `constructs.py`, structural SQL assertions):
- `fable_orchestrated_delegations` — `SELECT COUNT(*) FROM delegations WHERE orchestrator_model_family='anthropic' AND orchestrator_model='fable'` ≥ 3
- `multi_family_subagents` — `SELECT COUNT(DISTINCT subagent_model_family) FROM delegations WHERE orchestrator_model='fable'` ≥ 2
- `error_result_delegation` — `… WHERE result_status='error'` ≥ 1
- `unknown_result_delegation` — `… WHERE result_status='unknown'` ≥ 1
- `primary_model_family_populated` — `SELECT COUNT(*) FROM session_profiles WHERE primary_model_family IS NOT NULL` ≥ 3

---

## 6. Rebuild plan

All changes are **derived-tier only** — no durable migration, no backup manifest.
1. Edit canonical DDL: `session_profiles` (+2 cols), new `delegations` VIEW; bump `index.db` `SCHEMA_VERSION` 24 → 25.
2. Add `canonical_family()` to `core/sources.py`; wire `primary_model_*` into profile materialization.
3. Add `query_delegations` + `DelegationQueryRowPayload` + unit descriptor + card render profile.
4. Rebuild: `polylogue ops reset --index && polylogued run` (blue-green index replace; no upgrade helper — `devtools lab policy schema-versioning` forbids one for derived tiers).
5. New module under `polylogue/` (`demo/delegation.py`, `query/…delegation…`) ⇒ **regenerate topology projection**: `devtools render topology-projection && devtools render topology-status`, commit `docs/plans/topology-target.yaml` + `docs/topology-status.md`.
6. Regenerate `render openapi` + `render cli-output-schemas` (new payload model + unit) and `render cli-reference`/`render all --check` (grep for `out of sync`, don't trust the tail line).

---

## 7. Test strategy

- **VIEW correctness (unit):** delegation count == resolved non-quarantined `subagent` links; each field equals its structural source; `result_status='unknown'` exactly when no dispatch action or `is_error IS NULL`.
- **Construct-validity guard (property):** extend the `test_claim_vs_evidence` pattern — assert `result_status` is a pure function of `actions.is_error`/`exit_code`, never derivable from artifact text; fuzz artifact prose (inject "error"/"success" strings) and assert `result_status` unchanged.
- **DSL laws:** `test_query_exec_laws` / `test_verb_cardinality` / `test_query_fields` gain the `delegation` unit; `explain_query_expression` snapshot for the group-by-count example; `query_completions` offers the unit + fields.
- **Yield determinism:** on seed 1843, `yield_ratio` and the three cost buckets are exact fixed values; `unknown_cost` never folded into denominator.
- **Render:** delegation-card terminal + JSON snapshots (`__snapshots__`); `--explain` resolvability test (every line carries a block/profile ref).
- **Schema regen:** `cli-output-schemas` + `openapi` include `DelegationQueryRowPayload`; `render all --check` clean.
- **Demo:** new `DemoConstruct`s pass `demo verify`; `demo tour`/polydemo steps exit 0 within `FIRST_RESULT_BUDGET_S`/`FULL_TOUR_BUDGET_S`.
- **Non-injectivity guard:** `canonical_family` round-trip test — asserts the model→family map is total over demo models and that public DSL output emits **origin-vocab family**, not raw wire model tokens (Risk 1).

Verify via `devtools test <files>` (testmon-affected), not blanket runs.

---

## 8. Bead breakdown (proposed — do NOT create; 6 items, dependency-ordered)

1. **`primary_model_*` derived columns** (enabling primitive). AC: 2 cols on `session_profiles`; `canonical_family()` in `core/sources.py` total over demo models; populated by dominant-output-token rule; index bump 24→25; rebuild-plan doc; unit test for dominance + family mapping. *Blocks all others.*
2. **`delegations` VIEW.** AC: VIEW DDL merged; `result_status` derives only from `actions`; quarantined/prefix-sharing excluded; VIEW-correctness + construct-validity property tests green.
3. **`delegation` DSL unit + `query_delegations`.** AC: unit descriptor + payload model + field infos; `where | group by | count` lowers correctly; `explain_query_expression` snapshot; completions/cardinality/fields laws pass; openapi + cli-output-schemas regen. Dep: 2.
4. **Delegation-yield measure.** AC: group-by projection emits delegations/child_cost_total/ok_cost/wasted_cost/unknown_cost/yield_ratio with `cost_is_estimated` coverage footnote; `unknown` never in denominator; deterministic on seed 1843. Dep: 3.
5. **`delegation-card` render profile.** AC: profile in read-view registry, listed by `list_read_view_profiles`; plain+JSON+MCP snapshots; every card line `--explain`-resolvable; origin-vocab model projection. Dep: 2 (4 for the yield line).
6. **`delegation.polydemo` + seed/constructs.** AC: Fable multi-subagent fixture in seed.py (≥3 dispatches, ≥2 families, ≥1 error, ≥1 unknown, fixed costs); 5 new `DemoConstruct`s; executable demo (tour template) with iron-fist narration + resolving ROI finding + one honest "cannot answer" line; steps exit 0 within budget; topology projection regen. Dep: 3,4,5.

---

## 9. Top-3 risks

1. **Provider-vocab leak / model→family non-injectivity.** `messages.model_name` is a provider-wire token; `orchestrator_model_family`/`subagent_model_family` on a *public* DSL filter must project origin-vocab through the canonical `core/sources` map, exactly like the in-progress provider→origin retirement (where GEMINI+DRIVE collapse non-injectively into AISTUDIO_DRIVE). Ship raw wire model tokens on the public surface and you re-open bead polylogue-9e5.8's anti-goal. Mitigation: family mapping is the sole projection boundary, guarded by the non-injectivity round-trip test; keep raw `orchestrator_model`/`subagent_model` as provenance-only fields clearly distinct from `_family`.

2. **`result_status` construct validity — `unknown` is the majority for non-Claude.** Codex async subagents and any sidechain lacking a parent Task action have `is_error IS NULL` → `unknown`. If the yield denominator silently swallows `unknown`, ROI is a lie. Mitigation: hard-separate the `unknown_cost` bucket, exclude it from `yield_ratio`, render it explicitly on the card and as the demo's "cannot answer" slide. A delegation is gated on a real dispatch action, not merely `link_type='subagent'`.

3. **VIEW cost at 38 GB without the precompute.** Deriving dominant model per session inside the VIEW is O(messages) per delegation and would make `delegations where …` unusable on the live archive. Mitigation is structural: `primary_model_*` is precomputed on `session_profiles` (bead 1) so the VIEW is join-only; the `delegations`-VIEW-only path must never fall back to per-row message scanning. Secondary: the `actions` join keyed on `branch_point_message_id` relies on that column being populated at resolve time — unresolved links (parent not yet ingested) surface as `child_*` NULLs and must render as pending, not as zero-cost delegations.

No code or beads were modified. Key files for the implementer: `polylogue/storage/sqlite/archive_tiers/index.py` (VIEW + profile cols, `actions` at :324, `session_links` at :376), `polylogue/archive/query/metadata.py` (:740 unit descriptors), `polylogue/core/sources.py` (family map), `polylogue/demo/{constructs,seed,tour}.py`, `polylogue/storage/sqlite/queries/session_links.py` (edge lifecycle).

---

## [a41de797c7bc58dd8] SPEC — Temporal Doctrine (`docs/doctrine/time.md` + enforcement)

Grounding confirmed. Writing the spec. Key files read: `core/dates.py:37`, `core/timestamps.py`, `storage/sqlite/archive_tiers/index.py` (sessions DDL), `storage/sqlite/queries/filter_builder.py:88-94`, `archive/semantic/timing.py`, `tests/infra/frozen_clock.py`, beads `cpf`/`cpf.1`.

---

# SPEC — Temporal Doctrine (`docs/doctrine/time.md` + enforcement)

Wave-4 implementation-grade spec. Grounded in bead `polylogue-cpf` (the six-doctrine epic) and `polylogue-cpf.1` (TEXT-timestamp DDL lint). This spec is the **time doctrine** slice: the typed model, the total ordering comparator, the schema surface, the two live-bug fixes, and the enforcement lints.

## 0. Ground truth found in source (what already exists, what's wrong)

- Canon is already **UTC epoch-ms INTEGER** across durable tiers (`*_at_ms` columns everywhere in `archive_tiers/*.py`). The doctrine mostly *ratifies + guards* an existing convention; it does not migrate a TEXT regime.
- `sessions.sort_key_ms` is `INTEGER GENERATED ALWAYS AS (COALESCE(updated_at_ms, created_at_ms)) STORED` (`index.py:73`). **NULL when both source times are NULL** — a timeless session.
- `SessionTimingFacts.timing_provenance` already exists but is a **free-text string literal** `"sort_key_estimated"` (`timing.py:56,154`) — not typed, not enforced, only covers derived-duration facts, not the sort key or tz.
- **BUG 1** (`core/dates.py:37`): `parse_date` sets `"RELATIVE_BASE": datetime.now(tz=timezone.utc)` — a direct call on the real `datetime` symbol, not an injectable clock. `since:7d` resolves against uncontrolled wall-clock. `frozen_clock` can only reach it if every test remembers to register `polylogue.core.dates` via the `frozen_clock_modules` marker; there is **no single seam**, so relative-window queries are effectively non-hermetic.
- **BUG 2** (`queries/filter_builder.py:88-94`): `since`/`until` compile to `sort_key_ms >= ?` / `<= ?`. Since `sort_key_ms` is NULL for timeless sessions, `NULL >= x` is NULL/false → **timeless sessions silently vanish from every bounded window**. The `COALESCE(sort_key_ms, …, 0)` pattern in ordering paths (`archive_tiers/archive.py:1076`, `insights/session/rebuild.py:107`) compounds it: timeless rows get pinned at epoch-0 (1970), so they also sort to a wrong, invisible position rather than being surfaced as "time-unknown."

---

## 1. Schema additions

Placement rule (durability-keyed, per CLAUDE.md schema regimes): **all four new columns land on `index.db` (derived, rebuildable) — zero durable migration required.** They are all re-derivable from `source.db.raw_sessions` on rebuild. Bump `index.db` schema version, edit canonical DDL in `archive_tiers/index.py`, add the rebuild-plan note; do **not** write an upgrade helper (`lab policy schema-versioning` rejects it).

### 1.1 Typed time-kinds — `core/enums.py`

```python
class TimeKind(StrEnum):
    OCCURRED  = "occurred"   # provider-reported event wall-clock (created/updated/occurred_at_ms)
    ACQUIRED  = "acquired"   # when polylogue read the raw bytes (source.db, OUR clock)
    INGESTED  = "ingested"   # when the writer committed the derived row (parsed/materialized, OUR clock)
    SORT      = "sort"       # synthetic ordering key; NEVER a wall-clock claim
```

`OCCURRED` is the only tz-ambiguous, provider-trusted axis. `ACQUIRED`/`INGESTED` are always OUR injectable clock (§3.1) and are always known-UTC. `SORT` is derived and carries provenance (§1.2).

### 1.2 `sessions` (index.db) — four additive columns

```sql
-- sort_key_ms STAYS the generated COALESCE for the common case, BUT the writer
-- may override it (see §3.3): change to a plain written INTEGER column populated
-- by the writer, because the fell-back/synthesized tiers need a value the pure
-- COALESCE cannot express (ingest-time lives in a different tier).
sort_key_ms          INTEGER,                       -- was GENERATED; now writer-populated
sort_key_provenance  TEXT NOT NULL DEFAULT 'synthesized_zero'
                       CHECK({check("sort_key_provenance", SortKeyProvenance)}),
time_confidence      TEXT NOT NULL DEFAULT 'unknown'
                       CHECK({check("time_confidence", TimeConfidence)}),
source_tz            TEXT,                           -- IANA name or ±HH:MM offset; NULL = unknown
tz_provenance        TEXT NOT NULL DEFAULT 'unknown'
                       CHECK({check("tz_provenance", TzProvenance)}),
```

```python
class SortKeyProvenance(StrEnum):
    EXPLICIT             = "explicit"              # from updated_at_ms (provider-authored)
    INHERITED            = "inherited"             # fell back to created_at_ms
    FELL_BACK_TO_INGEST  = "fell_back_to_ingest"   # no event time; used acquired/ingested clock
    SYNTHESIZED_ZERO     = "synthesized_zero"      # no time anywhere; sentinel, MUST stay queryable

class TimeConfidence(StrEnum):
    EXACT     = "exact"      # hook-precise provider measurement
    REPORTED  = "reported"   # provider wall-clock, trusted as-is
    ESTIMATED = "estimated"  # inter-message-gap / sort_key_estimated derivation
    SYNTHETIC = "synthetic"  # no real time; ordering is a tiebreak sentinel only
    UNKNOWN   = "unknown"

class TzProvenance(StrEnum):
    PROVIDER_EXPLICIT = "provider_explicit"  # offset present in raw payload
    ASSUMED_UTC       = "assumed_utc"        # naive→UTC coercion (today's silent parse_timestamp behavior — now LABELED)
    INFERRED          = "inferred"           # derived from sibling signal (e.g. cwd/user profile)
    UNKNOWN           = "unknown"            # DEFAULT — tz-unknown-by-default
```

Doctrine invariants encoded by these types:
- **tz-unknown-by-default**: `tz_provenance` defaults `UNKNOWN`, `source_tz` defaults NULL. The existing `parse_timestamp` naive→UTC coercion is relabeled `ASSUMED_UTC` — an *honest* label for a guess, not a claim of truth.
- **Every `sort_key_ms` carries a `sort_key_provenance`.** A NULL/0 sort key is never anonymous; it is `SYNTHESIZED_ZERO` and MUST remain visible.
- Read payloads project `time_confidence` (§ `insights/registry.py` accessor + MCP/`read` payloads) so no consumer treats an `ESTIMATED` duration or `SYNTHETIC` ordering as ground truth.

---

## 2. The total, stable, skew-tolerant ordering comparator

**The transitivity trap (must not get wrong):** pairwise banding — "co-temporal if `|a−b| ≤ B` ⇒ equal" — is **not transitive** (a~b, b~c, but a≁c), so it cannot define a total order. The convergent fix is **quantization into equivalence classes**, then a fully deterministic identity tiebreak. Pairwise banding is used **only** for the read-payload `co_temporal` display flag, never for the sort.

Tuple: `(sort_key_ms, session_id, position, block_position)`. `B` = skew band in ms (default **2000**, config-surfaced).

```
# Total order over records r with quantized time axis.
# Rows with NULL sort_key_ms are SYNTHETIC: they sort into one defined bucket
# (after all timed rows) but remain PRESENT and identity-ordered — never dropped.

def time_bucket(r, B):
    if r.sort_key_ms is None:
        return (1, 0)              # (is_synthetic=1, bucket=0) -> sorts after all timed rows
    return (0, r.sort_key_ms // B) # floor-division => transitive equivalence classes

def compare(a, b, B):
    ba, bb = time_bucket(a, B), time_bucket(b, B)
    if ba != bb:
        return -1 if ba < bb else 1          # different quantum: time decides
    # same quantum (co-temporal within band) OR both synthetic: identity tiebreak.
    # This is total, antisymmetric, transitive, and STABLE across re-runs because
    # every component is content-identity, not wall-clock noise.
    for key in (a.session_id  vs b.session_id,
                a.position     vs b.position,       # NULL -> +inf sentinel
                a.block_position vs b.block_position):
        if key differs: return sign(key_a - key_b)
    return 0   # identical identity => genuinely equal (idempotent-write dedup guarantees uniqueness)

# Properties (property-tested, §4):
#   totality:      any a,b comparable
#   antisymmetry:  compare(a,b) == -compare(b,a)
#   transitivity:  holds because floor(x/B) is an equivalence relation (banding is NOT applied pairwise)
#   stability:     re-sorting the same corpus yields byte-identical order (no now()-dependence)
#   band-idempotence: perturbing any timestamp by <B within a bucket does not reorder
```

Co-temporal display flag (payload only, not sort): `co_temporal(a,b) = a.sort_key_ms is not None and b.sort_key_ms is not None and abs(a−b) ≤ B`.

**Monotonic-vs-wall separation:** the comparator consumes only `sort_key_ms` (wall/event time). Duration and latency facts (`timing.py`) are the *only* consumers permitted to use a monotonic delta; they must never be fed back into `sort_key_ms`. The doctrine states: wall time orders, monotonic time measures; the two never cross. `since`/`until`/`sort_key` are wall; `duration_ms`/latency percentiles are monotonic-class and carry `TimeConfidence.ESTIMATED`.

**Half-open intervals:** every bounded window is `[since, until)` — `since ≤ t < until`. This makes adjacent windows tile without double-counting (the `until <= ?` inclusive form in `filter_builder.py:93` is a defect: back-to-back `since:` windows double-count the boundary ms). Fix in §3.2.

---

## 3. Migration + the two live-bug fixes

### 3.1 New seam: `core/clock.py` (fixes BUG 1 hermetically)

Single injectable production clock — the one place `frozen_clock` patches.

```python
# core/clock.py
from datetime import datetime, timezone
def now(tz: timezone = timezone.utc) -> datetime: return datetime.now(tz)
def now_ms() -> int: return int(now().timestamp() * 1000)
def monotonic() -> float: import time; return time.monotonic()  # measurement-only, never for sort_key
```

`core/dates.py` fix:
```python
from polylogue.core import clock
settings = {..., "RELATIVE_BASE": clock.now()}   # was datetime.now(tz=timezone.utc)
```
`frozen_clock` gains one canonical registration: patch `polylogue.core.clock.now` / `.now_ms`. Any production module resolving "now" goes through this seam; the existing `verify-test-clock-hygiene` lint is extended to a **production** twin (§3.4) that bans direct `datetime.now`/`time.time` in production time-*resolution* paths (allowlist mirrors `docs/plans/test-clock-allowlist.yaml`).

### 3.2 Timeless-window fix (fixes BUG 2)

`filter_builder.py` `since`/`until` must (a) use half-open `[since, until)` and (b) **decide timeless rows explicitly** rather than dropping them via NULL-comparison:

```python
if since is not None:
    where_clauses.append("<col>.sort_key_ms >= ?")     # keep: NULL correctly excluded from a lower bound
    params.append(_iso_to_epoch(since) * 1000.0)
if until is not None:
    where_clauses.append("<col>.sort_key_ms < ?")      # was "<= ?": half-open [since, until)
    params.append(_iso_to_epoch(until) * 1000.0)
```

Timeless rows: add an explicit plan flag `include_timeless: bool` (default policy-driven). When a window is set and `include_timeless` is true, wrap: `(sort_key_ms >= ? [AND sort_key_ms < ?]) OR (sort_key_ms IS NULL AND ? )` where the last predicate is the doctrine choice — **timeless sessions are surfaced with a `time_confidence=SYNTHETIC` badge**, never silently vanished. Purge every `COALESCE(sort_key_ms, …, 0)` in *ordering* (`archive_tiers/archive.py:1076`, `insights/session/rebuild.py:107,111`, `threads.py:87`, `aggregates.py:40`): replace the 0-pin with an explicit `(sort_key_ms IS NULL)` leading sort term so timeless rows land in a *defined, visible* bucket instead of a fabricated 1970 position — matching the comparator's `time_bucket`.

### 3.3 sort_key de-generation + backfill (index.db, derived — rebuild, not migrate)

`sort_key_ms` becomes writer-populated (§1.2). Writer logic (`storage/repository` write path):
```
if updated_at_ms:  sort_key_ms, prov = updated_at_ms, EXPLICIT
elif created_at_ms: sort_key_ms, prov = created_at_ms, INHERITED
elif ingest_or_acquired_ms: sort_key_ms, prov = that, FELL_BACK_TO_INGEST   # ingest time is cross-tier -> pure COALESCE cannot do this
else:              sort_key_ms, prov = NULL, SYNTHESIZED_ZERO
```
Because index.db is rebuildable, the "migration" is: bump index schema version → `polylogue ops reset --index && polylogued run` repopulates all five columns from raw. **No numbered migration, no backup manifest.** Durable-tier care applies only if a future decision makes `source_tz`/`tz_provenance` durable (irreplaceable) — that path requires `migrations/{source|user}/NNN_*.sql`, one `PRAGMA user_version` step, behind a verified backup manifest; recommended answer is to keep them derived.

**Backfill invariant (doctrine-critical): a backfill must NEVER move an existing `sort_key_ms`.** Backfilling `sort_key_provenance`/`time_confidence`/`source_tz`/`tz_provenance` onto historical rows is a pure annotation. If a rebuild would compute a *different* `sort_key_ms` than the prior COALESCE value for a row that had one, that is a defect (it reorders history, invalidates keyset-pagination cursors and saved views). The rebuild recomputes provenance labels but pins `sort_key_ms` to the prior COALESCE result for any row where `updated_at_ms`/`created_at_ms` are unchanged.

### 3.4 Enforcement (lands the cpf hooks)

- **cpf.1 lint** (`devtools/schema_audit.py`, surfaced via `lab policy schema-versioning`): reject any new **durable-tier** DDL column whose name matches `*_at$|*_at_ms$|*_time$|*_ts$` typed as `TEXT` (must be `INTEGER` epoch-ms). Fixture: a TEXT-timestamp DDL fails; existing INTEGER `*_at_ms` pass. (AC lifted verbatim from `cpf.1`.)
- **Production clock-hygiene lint**: production twin of `verify-test-clock-hygiene` banning direct `datetime.now`/`time.time`/`time.monotonic` outside `core/clock.py` + allowlist.
- **Deny-lexicon tripwire fixture** (cpf's third cheap lint): a fixture asserting the doctrine text bans the folklore phrasings (e.g. bare `datetime.now()` in a timestamp comment, "sort_key is roughly the time") — the doctrine supersedes scattered time comments.
- `docs/doctrine/time.md` committed, linked from `docs/architecture-spine.md`; bd memory updated to point at the doctrine instead of restating time lore.

---

## 4. Test strategy

- **Hermetic clock:** all new time tests take `frozen_clock` and rely on the single `core/clock` seam (register `polylogue.core.clock` once, not N modules). Regression test for BUG 1: `find "since:7d"` under `frozen_clock` at `DEFAULT_FROZEN_EPOCH` returns a deterministic window boundary independent of host wall-clock; a second run after `frozen_clock.advance` shifts the boundary by exactly that delta. No `datetime.now` in the tested path (asserted by the production lint).
- **Backfill invariant (property, over `corpus_seeded_db`):** snapshot every `(session_id, sort_key_ms)`; run the provenance/tz/confidence backfill; assert `sort_key_ms` is byte-identical for all rows that kept their source times. Only the new annotation columns may change.
- **Comparator property tests (Hypothesis):** totality, antisymmetry, transitivity (the quantization guarantee), stability (sort twice → identical), band-idempotence (perturb any timestamp by `< B` within a bucket → order unchanged), and NULL-sort-key rows are always *present and last, never dropped*.
- **Timeless-window regression (BUG 2):** seed a session with `created_at_ms = updated_at_ms = NULL`; assert it is (a) excluded from a lower-bounded `since:` window's *timed* set but (b) surfaced with `time_confidence=SYNTHETIC` when `include_timeless`, and (c) never assigned a 1970 position by any ordering path.
- **Half-open tiling:** two adjacent windows `[t0,t1)` + `[t1,t2)` over a corpus contain each boundary-ms row exactly once (no double-count).
- **Lint fixtures:** cpf.1 TEXT-timestamp DDL rejection; production clock-hygiene violation rejection; deny-lexicon tripwire.
- Protected-file discipline: extend, don't replace, `tests/unit/core/test_properties.py`.

---

## 5. Bead breakdown (children of `polylogue-cpf`; `cpf.1` already exists)

1. **`time.doctrine`** — Write `docs/doctrine/time.md` (four TimeKinds, UTC epoch-ms canon, skew-band quantization, `[since,until)`, monotonic-vs-wall, tz-unknown-by-default, backfill-never-moves-sort_key), link from architecture-spine, retire scattered time comments. **AC:** doc committed + indexed; spine links it; ≥3 folklore comments repointed; deny-lexicon fixture green.
2. **`time.clock-seam`** — Add `core/clock.py`; route `core/dates.py` through it; production clock-hygiene lint + allowlist. **AC:** `since:7d` hermetic under `frozen_clock` (regression test); lint fails on a direct `datetime.now` fixture; no direct `datetime.now` remains in time-resolution paths.
3. **`time.enums`** — Add `TimeKind`/`SortKeyProvenance`/`TimeConfidence`/`TzProvenance` to `core/enums.py`; wire `check(...)` CHECKs; regenerate `render openapi` + `render cli-output-schemas`. **AC:** enums embedded in generated surfaces; `render all --check` clean (grep `out of sync`).
4. **`time.schema`** — index.db: de-generate `sort_key_ms` to writer-populated + add 4 columns; writer provenance logic; rebuild-plan note; index schema bump. **AC:** fresh `ops reset --index && polylogued run` populates all five; no numbered migration introduced; `lab policy schema-versioning` green.
5. **`time.comparator`** — Implement quantized total comparator + `co_temporal` flag; replace `COALESCE(sort_key_ms,…,0)` ordering pins with explicit `IS NULL` buckets. **AC:** property tests (totality/antisymmetry/transitivity/stability/band-idempotence) green; timeless rows present-and-last everywhere.
6. **`time.windows`** — `filter_builder` half-open `[since,until)` + `include_timeless` plan flag + `time_confidence` on read payloads. **AC:** BUG-2 regression green; half-open tiling test green; timeless sessions surfaced not vanished.
7. **`cpf.1`** (exists) — TEXT-timestamp durable-DDL lint. **AC:** unchanged from bead.
8. **`time.backfill-guard`** — Property test asserting backfill never moves an existing `sort_key_ms` over `corpus_seeded_db`. **AC:** snapshot-equality invariant green; wired into `devtools verify` affected-set.

(6 net-new beads 1–6+8 plus the pre-existing `cpf.1`; ships as one coherent phase on `feature/doctrine/time`, commits as review waypoints.)

---

## 6. Top-3 risks

1. **De-generating `sort_key_ms` reorders history / breaks cursors.** Turning a generated column into a writer-populated one risks the writer computing a value that differs from the old `COALESCE(updated_at_ms, created_at_ms)` for existing rows — silently reordering the archive, invalidating keyset-pagination cursors (`idx_messages_session_sortkey`, `#2467/#2475`) and saved views. **Mitigation:** the backfill-invariant property test (bead 8) is a hard gate; writer pins `sort_key_ms` to the prior COALESCE result whenever source times are unchanged, and only the FELL_BACK_TO_INGEST/SYNTHESIZED tiers (previously NULL) get new values.
2. **Skew-band non-transitivity if implemented as pairwise.** If a future contributor "simplifies" the quantization back to pairwise `|a−b| ≤ B ⇒ equal`, the order stops being transitive and sorts become unstable/non-deterministic (a latent, corpus-dependent bug). **Mitigation:** the transitivity + stability property tests fail loudly; the doctrine text explicitly separates quantization (sort) from banding (display flag).
3. **BUG-1 fix incomplete if `dateparser` or a caller re-captures wall-clock.** Routing `RELATIVE_BASE` through `core/clock` fixes `parse_date`, but any other relative-time resolver (or a memoized base) that calls `datetime.now`/`time.time` directly re-opens the non-hermetic hole. **Mitigation:** the production clock-hygiene lint (bead 2) makes the seam mandatory repo-wide, not just at the one known call site; allowlist is explicit and reviewed.

**Not run** (spec-only, no code/beads edits per mandate): no tests executed, no DDL applied, no `render` invoked. All file/line citations are absolute-path-relative to `/realm/project/polylogue/`.

---

## [a2f4e84c9b44c6209] SPEC — Attachment referenced-vs-stored-bytes honesty (83u / 83u.4 / 83u.6 / #2468)

I have enough grounding. Key confirmed facts before writing:

- The `attachments` DDL (index.db, `index.py:519`) already carries `byte_count`, `blob_hash BLOB` (NULL unless acquired), `acquisition_status ∈ {acquired,unavailable,unfetched}`, `ref_count`. **No new column is needed for the honesty split.**
- Every read query (`attachment_records.py:get_attachments*`, `rebuild.py:sync_attachment_batch`, `write.py:1184`) selects `a.byte_count AS size_bytes` but **drops `acquisition_status` and `blob_hash`**. So `AttachmentRecord` (`records.py:190`) and the public `Attachment` (`archive/attachment/models.py`) expose only `size_bytes` — every consumer reads declared bytes as retrievable. That is the red-team gap, and it is a projection defect, not a schema defect.
- A grouped classifier already exists: `MissingReferencedBlobReport` in `storage/blob_integrity.py` (by table/ref_type/origin) but it operates on source-tier refs, not index-tier attachment acquisition state, and unfetched NULL-hash rows must not be counted as missing.

---

# SPEC — Attachment referenced-vs-stored-bytes honesty (83u / 83u.4 / 83u.6 / #2468)

## 0. Thesis

The archive already stores the *truth* (`acquisition_status`, nullable `blob_hash`) but **projects a lie**: read surfaces emit `size_bytes = byte_count` with no acquisition qualifier, so coverage/cost/insight/MCP consumers sum *referenced* bytes as if *stored/retrievable*. The fix is a projection + accounting discipline, not a data-model change: **split `referenced_bytes` (declared, from `byte_count`) from `stored_bytes` (on-disk, only where `acquisition_status='acquired'` and the blob file exists), and never let a surface sum referenced bytes into a "retrievable" total.** Because the DDL is already honest and index.db is a *derived* tier, this needs **no numbered migration and no schema bump** — only DDL-surfacing of existing columns through the read path, plus one census/classifier and a citation-anchor resolver.

---

## 1. Schema / read-surface changes

### 1a. No new index column (grounding correction)
`attachments` already has the three load-bearing columns. Do **not** add a `referenced_bytes`/`stored_bytes` pair to the table — that would duplicate `byte_count` and re-introduce a drift surface. `referenced_bytes := byte_count`; `stored_bytes` is *computed on read* as `byte_count` gated by `acquisition_status='acquired' AND blob file present`.

### 1b. Propagate `acquisition_status` + `blob_hash` through read queries
Add to the SELECT lists of the four attachment read paths (currently dropping them):
- `storage/sqlite/queries/attachment_records.py::get_attachments`, `get_attachments_batch`
- `storage/insights/session/rebuild.py::sync_attachment_batch`
- `storage/sqlite/archive_tiers/write.py:1184` (the sync-batch read)

New projected columns: `a.acquisition_status`, `a.blob_hash`.

### 1c. New closed enum + fields on `AttachmentRecord` (`storage/runtime/archive/records.py:190`) and public `Attachment` (`archive/attachment/models.py`)
```
AttachmentEvidence = Literal["stored", "unfetched", "unavailable"]   # core/enums.py
```
Replace the bare `size_bytes: int | None` semantics with an explicit trio (keep `size_bytes` for byte-compat, add):
- `referenced_bytes: int` = `byte_count` (always the declared count; the number the source claims)
- `stored_bytes: int` = `byte_count if evidence=='stored' else 0`
- `evidence: AttachmentEvidence` derived from `acquisition_status` (`acquired→stored`, `unfetched→unfetched`, `unavailable→unavailable`)
- `blob_hash_hex: str | None` (present only when `evidence=='stored'`)

`size_bytes` is retained but **must be documented as referenced-not-retrievable** and every *new* consumer reads `referenced_bytes`/`stored_bytes` explicitly.

### 1d. Read-payload projection rule (the honesty invariant)
At every aggregate boundary that today emits a single attachment byte figure — session profile (`SessionEvidencePayload.attachment_count` → add `attachment_referenced_bytes` + `attachment_stored_bytes`), stats (`queries/stats.py`), coverage, cost, and the MCP insight payloads (`insights/registry.py` descriptors) — emit **two named sums, never one**:
```
attachment_referenced_bytes  (SUM byte_count over all refs)
attachment_stored_bytes      (SUM byte_count WHERE evidence='stored')
```
Invariant enforced by a new lint (`devtools lab policy`): no surface may emit an attachment byte total keyed with a bare `bytes`/`size`/`total_bytes` name; it must be one of the two qualified names. This is the mechanical guard against a future regression re-fusing the two.

---

## 2. Algorithms (pseudocode)

### 2a. Evidence derivation (single source of truth)
```
def attachment_evidence(acquisition_status, blob_hash, blob_present) -> AttachmentEvidence:
    if acquisition_status == 'acquired':
        # honesty floor: an 'acquired' row whose file is gone is NOT stored
        return 'stored' if (blob_hash is not None and blob_present) else 'unavailable'
    if acquisition_status == 'unavailable':
        return 'unavailable'
    return 'unfetched'          # blob_hash MUST be NULL here (assert; never synthetic)
```
`blob_present` resolved via `storage/blob_store.py` path helper, `mode=ro`. In hot read paths where per-blob stat is too costly, `stored_bytes` may trust `acquisition_status='acquired'` and the census (2c) reconciles the "acquired-but-file-missing" delta out-of-band; the citation-anchor (2d) always does the stat.

### 2b. Two-sum aggregation (replaces any single SUM(byte_count))
```
referenced_bytes = Σ byte_count over refs in scope
stored_bytes     = Σ byte_count where evidence == 'stored'
# NEVER: retrievable_bytes = referenced_bytes   ← the banned line
unfetched_bytes  = Σ byte_count where evidence == 'unfetched'
unavailable_bytes= Σ byte_count where evidence == 'unavailable'
assert referenced_bytes == stored + unfetched + unavailable   # partition invariant
```

### 2c. Byte census (83u.6) — read-only, `mode=ro`, no write conn
```
open index.db (mode=ro); open source.db (mode=ro) only for source_ref classification
for each (origin, acquisition_status) group over attachments⋈attachment_refs:
    attachment_count, ref_count
    declared_byte_sum          = Σ byte_count
    acquired_blob_count        = count(acquisition_status='acquired' AND blob_hash NOT NULL)
    acquired_blob_bytes_on_disk = Σ stat(blob_path).size  where file exists   # true on-disk
    unfetched_count, unavailable_count
    missing_blob_ref_count     = acquisition_status='acquired' AND file absent  # the floor breach
    top source_ref classes (upload_origin, source_url host, native_id kind)   # for classification
emit permanent_unfetchable_floor = classify(unfetched ∪ unavailable) → see 2c-classify
persist JSON + short markdown under .agent/scratch/research/ (per 83u.6 AC)
reconcile totals against `ops diagnostics workload --blob-reference-debt --json`
```
`missing_blob_ref_count` must be **0** in the healthy state; any nonzero is a floor breach (acquired row lost its file) and is reported as *acquisition debt*, distinct from `unfetched` honest-absence (83u.4 AC 2/3).

### 2c-classify. Missing-referenced-blob classification (83u.4)
Extend `MissingReferencedBlobReport` (`blob_integrity.py`) OR add an attachment-tier sibling so the two debts are **separately reported**:
```
for each attachment ref with no stored blob:
    if acquisition_status == 'unfetched':  →  bucket = honest_absent  (NOT missing debt)
    elif live_handle_reachable(ref):       →  bucket = reacquirable   (feeds 83u.2/83u.3 as bug)
        # drive/oauth handle, source_url resolvable, local path under allowlist
    else:                                  →  bucket = permanently_gone (source deleted / pre-install / provider-expiry)
report(bucket) with (table, ref_type, origin, count, sample≤20)
# unfetched NULL-hash rows are honest-absent, never 'missing_referenced_blobs'
```
`live_handle_reachable` is a *classification hint only* (upload_origin + source_url shape + local-path allowlist membership), not a fetch — the actual re-acquisition is 83u.2's job.

### 2d. Citation-anchor resolver (the anti-confident-hit)
A resolver used by any surface that cites an attachment as evidence (`resolve_ref`, postmortem bundles, recall packs):
```
def resolve_attachment_citation(attachment_id) -> AttachmentCitation:
    row = SELECT byte_count, acquisition_status, blob_hash FROM attachments WHERE id=?
    ev  = attachment_evidence(row.acquisition_status, row.blob_hash, blob_present=stat(path))
    match ev:
      'stored'      → Citation(status='exists', bytes=byte_count, blob_hash=hex, retrievable=True)
      'unfetched'   → Citation(status='unfetched', referenced_bytes=byte_count, retrievable=False,
                                note='referenced by source; bytes never fetched')
      'unavailable' → Citation(status='permanently_gone', referenced_bytes=byte_count, retrievable=False,
                                note='source no longer holds these bytes')
    # a citation NEVER returns a confident hit with bytes for a non-stored attachment
```
Anchor renders as `"exists N bytes"` / `"unfetched (N bytes referenced, not stored)"` / `"permanently gone"` — three states, never a bare byte figure implying retrieval.

---

## 3. Migration

- **Index tier is derived → no numbered migration, no `PRAGMA user_version` bump.** The columns already exist (added when `blob_hash` became honest-nullable + `acquisition_status` at index v13). Per CLAUDE.md schema-regime rules, a read-projection change edits canonical DDL only if a column changes — here **none does**.
- **No durable-tier (`source.db`/`user.db`) change.** No backup manifest required.
- **Deployment = code-only.** New model fields are additive and defaulted; existing archives read correctly with no rebuild. A rebuild (`ops reset --index && polylogued run`) is **not** required, but the census (83u.6) should be run once as the "before" baseline immediately after deploy.
- **`render` regeneration required:** new `AttachmentEvidence` enum + new payload fields flow into `render openapi` + `render cli-output-schemas` + `render topology-*` (if a module is added for the census). Run `devtools render all --check` and grep for `out of sync`.

---

## 4. Test strategy

1. **Evidence-derivation unit** (`core/enums` + helper): table-driven over the 3×{blob present/absent}×{hash null/nonnull} matrix; asserts `acquired+file-absent → unavailable`, `unfetched+nonnull-hash → assertion error` (synthetic-hash guard).
2. **Two-sum partition property** (Hypothesis, in `tests/property`): for any seeded mix of statuses, `referenced == stored + unfetched + unavailable`, and `stored ≤ referenced`. Guards against a surface that sums wrong.
3. **Read-projection contract** (`tests/unit/storage`): seed one session with one acquired + one unfetched + one unavailable attachment; assert `get_attachments` / `sync_attachment_batch` return `evidence` and `stored_bytes==0` for the two non-stored rows, `referenced_bytes==byte_count` for all three.
4. **Honesty lint** (`tests/unit/devtools` + a `devtools lab policy` check): scan insight/coverage/MCP payload schemas; fail if any attachment byte field is named with a bare `bytes`/`size`/`total` (must be `*_referenced_bytes`/`*_stored_bytes`). This is the durable regression guard — the red-team gap re-appears the moment a single-figure field is added.
5. **Census reconciliation** (`tests/unit/storage`, seeded, not live): census totals equal the direct SQL two-sums; `missing_blob_ref_count==0` on a clean seed; inject one acquired-file-missing row → census reports it as acquisition debt, **not** as `unfetched`, and **not** in `missing_referenced_blobs` for unfetched rows.
6. **Citation-anchor snapshot** (`tests/unit/mcp` or resolve_ref test): the three states render three distinct strings; assert no non-stored citation carries `retrievable=True` or an unqualified byte figure.
7. **Live read-only smoke** (manual, not CI): run census against `POLYLOGUE_ARCHIVE_ROOT=/home/sinity/.local/share/polylogue` with `mode=ro`, cross-check against `ops diagnostics workload --blob-reference-debt --json` (83u.6 AC verify). No write conn opened.

Use `frozen_clock`; verify via `devtools test <files>` (never blanket dir).

---

## 5. Bead breakdown (children under 83u, with acceptance)

**B1 — Surface acquisition evidence through attachment read models** *(size:S, foundation; blocks the rest)*
Add `AttachmentEvidence` enum; project `acquisition_status`+`blob_hash` through the 4 read queries; add `referenced_bytes`/`stored_bytes`/`evidence`/`blob_hash_hex` to `AttachmentRecord` + public `Attachment`; derive via 2a.
*AC:* a seeded session with one acquired + one unfetched + one unavailable attachment returns `evidence` correctly and `stored_bytes==0` for non-stored; `size_bytes` retained + docstring'd as referenced-not-retrievable; mypy-strict green.

**B2 — Two-sum accounting on every aggregate surface** *(size:M; depends B1)*
Replace any single attachment byte figure in session profile, stats, coverage, cost, and MCP insight payloads with the qualified `*_referenced_bytes` + `*_stored_bytes` pair; add the partition assertion.
*AC:* no surface emits a bare attachment byte total; profile/coverage/cost/MCP each expose both sums; partition property test passes; `render openapi`+`cli-output-schemas` regenerated and in sync.

**B3 — Honesty lint (durable regression guard)** *(size:S; depends B2)*
`devtools lab policy` check forbidding bare-named attachment byte fields; wire into `devtools verify`.
*AC:* lint fails on a deliberately-added `attachment_bytes` field and passes on the qualified pair; documented in `docs/devtools.md`.

**B4 — Byte census by origin × status (83u.6)** *(size:M; depends B1)*
Read-only census command (2c), JSON+markdown artifact under `.agent/scratch/research/`, reconciles against blob-reference-debt.
*AC (from 83u.6):* grouped `(origin, acquisition_status)` with attachment_count/declared_byte_sum/acquired_blob_count/bytes-on-disk/unfetched/unavailable/missing_blob_ref_count; runs `mode=ro` against the live archive with no write conn; totals reconcile with `ops diagnostics workload --blob-reference-debt --json`; before-baseline captured and written to parent 83u.

**B5 — Missing-referenced-blob classifier: attachment-tier vs source-tier split (83u.4)** *(size:M; depends B4)*
Extend/sibling `MissingReferencedBlobReport` so unfetched NULL-hash rows are `honest_absent` (never missing debt); classify the remainder into `reacquirable` / `permanently_gone` (2c-classify); update backup warning wording to say *source-tier* referenced blobs.
*AC (from 83u.4):* diagnostic reports source-tier backup debt separately from index-tier attachment acquisition state; unfetched rows not counted as missing; acquired-file-absent rows counted as attachment acquisition debt; backup warning/docs/tests updated; classification table (bucket, ref_type, origin, count, sample).

**B6 — Citation-anchor resolves attachment evidence, not a confident hit** *(size:S; depends B1)*
Resolver (2d) used by `resolve_ref` / postmortem / recall surfaces; three-state rendering.
*AC:* an attachment citation renders `exists N bytes` only when `evidence='stored'` with a present blob; `unfetched` and `permanently_gone` render explicit non-retrievable strings with the referenced byte count labelled as referenced; no non-stored citation carries `retrievable=True`; snapshot test covers all three.

*(Optional B7 — permanent-unfetchable-floor doc, size:S: fold the census's `permanently_gone` bucket into a short honesty note in `docs/internals.md` so "attachments preserved" claims cite the floor. Defer unless operator wants it.)*

Ordering: **B1 → {B2→B3, B4→B5, B6}**. B1 is the single foundation; B4/B5 (census+classifier) and B2/B3 (accounting+lint) and B6 (citation) then parallelize.

---

## 6. Top-3 risks

1. **`size_bytes` retained for byte-compat becomes the silent leak.** Every existing consumer still reads `size_bytes` and treats it as retrievable; adding `stored_bytes` alongside doesn't fix them unless call sites migrate. *Mitigation:* the B3 honesty lint must flag *reads* of `size_bytes` in aggregate contexts, not only field names — or, stronger, rename `size_bytes`→`referenced_bytes` at the model and let mypy-strict surface every call site (preferred; the CLAUDE.md note calls mypy the primary net for identifier refactors). Ship the rename in B1 rather than keeping a compat alias.

2. **Per-blob `stat()` in hot read paths.** Deriving `evidence='stored'` truthfully requires confirming the blob file exists (2a); doing this per-attachment in `get_attachments_batch` over large sessions adds an I/O storm. *Mitigation:* in hot paths trust `acquisition_status='acquired'` for `stored_bytes` and let the census (B4) reconcile the acquired-but-file-missing delta as a bounded batch job; only the citation-anchor (B6) and census pay the stat cost. Document the two trust levels explicitly so a reviewer doesn't "fix" the hot path into a storm.

3. **Classifier miscounts unfetched as debt (the exact bug 83u.4 warns against).** The existing `missing_referenced_blobs` counter operates on source-tier refs; naively unioning attachment refs would count 6,414 honest-unfetched NULL-hash rows as "missing," recreating the fabricated 39,586-style warning. *Mitigation:* B5's first assertion is `acquisition_status='unfetched' ⇒ honest_absent`, with a regression test that injects unfetched rows and asserts they never enter `missing_*`; keep index-tier attachment debt and source-tier backup debt in separate report sections (83u.4 AC 1/4).

---

**Convergent-design notes:** (a) construct-validity — the census's `permanent_unfetchable_floor` is reported as *normal baseline accounting* (source-deleted/pre-install/provider-expiry per the 83u operator reframe), never as a defect backlog, so "attachments preserved" claims are measured against an honest denominator. (b) citation-anchor — B6 makes attachment evidence a first-class three-state resolution (`exists N bytes` / `unfetched` / `permanently gone`), the same anchor discipline used for other evidence refs, so no downstream surface can cite referenced bytes as a confident retrievable hit.

Files that must change (all absolute): `/realm/project/polylogue/polylogue/storage/sqlite/queries/attachment_records.py`, `/realm/project/polylogue/polylogue/storage/insights/session/rebuild.py`, `/realm/project/polylogue/polylogue/storage/sqlite/archive_tiers/write.py`, `/realm/project/polylogue/polylogue/storage/runtime/archive/records.py`, `/realm/project/polylogue/polylogue/archive/attachment/models.py`, `/realm/project/polylogue/polylogue/core/enums.py`, `/realm/project/polylogue/polylogue/storage/blob_integrity.py`, `/realm/project/polylogue/polylogue/insights/archive_models.py` + `registry.py`, `/realm/project/polylogue/polylogue/storage/sqlite/queries/stats.py`. DDL (`/realm/project/polylogue/polylogue/storage/sqlite/archive_tiers/index.py:519`) is **unchanged** — the columns already exist.

---

## [a02cb24394f76da25] Wave-5 Spec: secret-redaction-at-read + verifiable-forget (`kwsb` / `27m` / `jnj.5`)

I have the grounding I need. Delivering the spec.

---

# Wave-5 Spec: secret-redaction-at-read + verifiable-forget (`kwsb` / `27m` / `jnj.5`)

## Grounding (verified in source)

- `core/hashing.py` — `hash_text` (NFC + SHA-256), `hash_payload` (sorted-keys JSON). `pipeline/ids.py:session_content_hash` hashes title/timestamps/messages/blocks/attachments/events, **excludes user metadata by construction**.
- **Idempotency skip is the resurrection vector**: `ingest_batch/_core.py:400` `content_unchanged = existing_hash_hex == payload.content_hash` → skip; `_core.py:475` if `existing_row is None` it *writes*. So re-acquiring the original source file (same bytes → same parse → same `content_hash`) rewrites an excised row unless intercepted.
- `source.db` v2 (`archive_tiers/source.py`): `raw_sessions.blob_hash` → content-addressed blob; `blob_refs` is reference-counted; **no tombstone/redacted column exists**. `history_sidecars` already carries a `content_hash BLOB(32)` precedent.
- `blob_gc.py` — lease-safe, reference-counted delete keyed on `blob_refs`/`pending_blob_refs`; dropping a ref → GC reclaims.
- `index.db` — `sessions.content_hash BLOB(32)`; messages→blocks→attachments cascade `ON DELETE CASCADE`; FTS is contentless, kept in sync by triggers (`index.py:301,306`). **`embeddings.db` is a separate tier with no cross-DB FK** → must be deleted explicitly by message ref.
- `AssertionKind` (`enums.py:399`) is closed but stored as `TEXT` (no CHECK) — a new value needs no user-tier migration, but is embedded in `render openapi` + `render cli-output-schemas` (regen gotcha). `AssertionStatus` already has `CANDIDATE/ACCEPTED/REJECTED` → the judgment queue is free.
- **jnj.5 confirmed**: `reset.py:311-339` — `--session`/`--source` call `_tombstone_archive_sessions` (writes `upsert_suppression`) behind only a bypassable `click.confirm`; no dry-run of *resolved* targets; a typo falls through `_resolve_archive_session_ids:175` (`resolved.append(token)`) and writes a suppression for a nonexistent id.
- MCP `delete_session` (`server_mutation_tools.py:753`) → `delete_session_safe` (`api/archive.py:4327`) → `archive.delete_sessions` — **no audit row, `confirm:bool` only**, a different contract than reset.
- Read boundary: `get_messages` (`repository/archive/sessions.py:113`, MCP `server_tools.py:854`); read views already carry `no_redact`/`redact_paths` (`read_views/base.py:51`, `context.py:38,80`) — the hook point exists.
- `docs/plans/security-privacy-coverage.yaml` exists; kwsb owns its gaps.

---

## 1. Schema / DDL + tier + regime

**A. `excision_tombstones` — source.db (durable, v2→v3, additive numbered migration under backup manifest)**

```sql
CREATE TABLE IF NOT EXISTS excision_tombstones (
    tombstone_id           TEXT PRIMARY KEY,           -- uuid
    session_id             TEXT NOT NULL,              -- origin:native_id (logical, not FK — survives rebuild)
    scope                  TEXT NOT NULL CHECK(scope IN ('session','message','block','span')),
    target_message_id      TEXT,
    target_block_id        TEXT,
    span_start             INTEGER,                    -- char offsets; coords only, never content
    span_len               INTEGER,
    original_content_hash  BLOB NOT NULL CHECK(length(original_content_hash)=32), -- session hash BEFORE excision (idempotency key)
    excised_content_hash   BLOB NOT NULL CHECK(length(excised_content_hash)=32),  -- session hash AFTER excision
    original_blob_hash     BLOB CHECK(original_blob_hash IS NULL OR length(original_blob_hash)=32), -- scrubbed raw blob, audit only
    span_text_hash         BLOB CHECK(span_text_hash IS NULL OR length(span_text_hash)=32),         -- hash(removed text) — reconciliation probe
    reason                 TEXT NOT NULL,
    excised_at_ms          INTEGER NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS idx_tombstone_original_hash ON excision_tombstones(original_content_hash);
CREATE INDEX IF NOT EXISTS idx_tombstone_session ON excision_tombstones(session_id);
```

Rationale for a **separate table** (not columns on `raw_sessions`): a session can have N tombstones; keeps the additive migration minimal; `session_id` is deliberately **logical, not FK** — same reasoning as `branch_point_message_id` in CLAUDE.md (a rebuild/full-replace DELETE would orphan an FK). Two hashes is the crux: `original_content_hash` is the re-ingest idempotency key; `excised_content_hash` is what the archive now holds.

**B. `mutation_audit` — ops.db (disposable, v1; edit canonical DDL, no migration chain)**

```sql
CREATE TABLE IF NOT EXISTS mutation_audit (
    audit_id           TEXT PRIMARY KEY,
    operation          TEXT NOT NULL,      -- excise|reset_session|reset_source|reset_database|delete_session|mcp_admin_delete
    actor              TEXT NOT NULL,      -- cli|mcp|daemon
    target_ref         TEXT NOT NULL,      -- as typed
    target_kind        TEXT NOT NULL,      -- session|source_path|scope
    resolved_targets_json TEXT NOT NULL DEFAULT '[]', -- resolved [{origin,native_id,counts}]
    affected_tiers_json   TEXT NOT NULL DEFAULT '[]', -- [source,index,fts,embeddings,blob]
    reason             TEXT,
    old_content_hash   BLOB, new_content_hash BLOB,
    dry_run            INTEGER NOT NULL CHECK(dry_run IN (0,1)),
    target_count       INTEGER NOT NULL DEFAULT 0,
    requested_at_ms    INTEGER NOT NULL,
    applied_at_ms      INTEGER            -- NULL for dry-run / aborted
) STRICT;
```

**C. `AssertionKind.SECRET_CANDIDATE = "secret_candidate"`** — user.db (durable v4, **no migration**: TEXT column). Candidate rows use `AssertionStatus.CANDIDATE`; `context_policy_json` default `{"inject": false}`. Body stores **span coordinates + detector rule id + entropy score only — never the matched bytes**. Regenerate `render openapi` + `render cli-output-schemas` (enum-embed gotcha).

Regime summary: source.db → **additive numbered migration `003_excision_tombstones.sql` + verified backup manifest**; ops.db → edit DDL (rebuilt on reset); index/embeddings → **no DDL change** (excision deletes rows + rebuilds the session via existing blue-green machinery); user.db → enum value only.

---

## 2. Scan + excision algorithms (pseudocode)

**Secret scan (ingest-side + retro + read-boundary), value never persisted:**

```
scan_blocks(session) -> list[Candidate]:
    for block in session.blocks:
        text = block.search_text                       # already materialized
        for rule in RULESET:                           # gitleaks-class regex: aws/gh/slack/pk/jwt/…
            for m in rule.finditer(text):
                yield Candidate(session_id, message_id, block_id,
                                span_start=m.start(), span_len=len(m.group()),
                                rule_id=rule.id, entropy=shannon(m.group()))
        for tok in high_entropy_tokens(text, min_len=20, min_bits=3.5):
            yield Candidate(..., rule_id="entropy", entropy=tok.bits)
    # INVARIANT: Candidate carries NO m.group() text — coordinates + rule id + score only
```

- **Ingest path**: after materialize, emit `SECRET_CANDIDATE` assertions (`status=candidate`). Registration trap: assert kind present before write.
- **Retro command** `polylogue ops scan-secrets [--since] [--rate N/s]`: bounded pass over the corpus.
- **Read-boundary overlay** (non-mutating): in `get_messages`/read-view render, run `scan_blocks` on the returned blocks, **mask** matched spans (`••• secret_candidate:aws_key •••`) unless a gated `--reveal-secrets` (CLI) / `reveal=true` (MCP) / explicit web toggle is set. Bytes stay durable; reveal is a read affordance, not a mutation. Reveal writes a `mutation_audit`-adjacent read-audit row (optional) but **never logs the value**.

**Excision (`polylogue ops excise <ref> --reason … [--dry-run|--yes]`):**

```
excise(ref, scope, reason, dry_run):
    session = resolve(ref)                              # ambiguous prefix -> error (reuse reset resolver, fixed)
    plan = MutationPlan(operation="excise", targets=[…], affected_tiers=[source,index,fts,embeddings,blob])
    old_hash = index.sessions[session].content_hash
    scrubbed = replace_span_with_marker(parsed_session, scope)   # "[[excised: <tombstone_id>]]"
    new_hash = session_content_hash(scrubbed)
    plan.old_content_hash, plan.new_content_hash = old_hash, new_hash
    print_dry_run(plan)                                # resolved targets + counts + tiers, BEFORE any write
    record_mutation_audit(plan, dry_run=True)
    if dry_run or not yes: return                      # --yes gate BEFORE first mutation (jnj.5 contract)
    with source_txn, index_txn:                        # single writer
        # 1. source: rewrite stored raw blob (scrub span), repoint raw_sessions.blob_hash, drop old blob_ref
        new_blob = scrub_bytes(load_blob(raw.blob_hash), span); old_blob = raw.blob_hash
        raw.blob_hash = put_blob(new_blob); drop_blob_ref(old_blob, raw_id)   # GC reclaims secret-bearing blob
        # 2. tombstone
        insert excision_tombstones(original_content_hash=old_hash, excised_content_hash=new_hash,
                                   original_blob_hash=old_blob, span_text_hash=hash_text(removed), scope, reason)
        # 3. index: rebuild the ONE session from scrubbed parse (blue-green session rebuild)
        rebuild_session(scrubbed)  # cascades blocks+FTS via triggers/ON DELETE CASCADE
        # 4. embeddings: explicit delete by message/block ref (no cross-db FK)
        embeddings.delete_where(session_id=session)    # converger re-embeds scrubbed text
        # 5. mark accepted candidate assertions accepted; reject-path -> suppression assertion
    record_mutation_audit(plan, dry_run=False, applied_at_ms=now)
```

**Idempotency resurrection guard (ingest-time, `_core.py` before the skip/write branch):**

```
tomb = source.excision_tombstones.by_original_hash(payload.content_hash)
if tomb:                                     # this source STILL contains the secret
    scrubbed = re_apply_excision(payload.parsed_session, tomb)   # re-scrub raw blob too
    write scrubbed (content_hash == tomb.excised_content_hash); do NOT surface span
    return   # idempotency now settles on the excised hash, never the original
```

This is the *"re-ingesting the original source does not resurrect it"* AC.

---

## 3. Migration (durable-tier care)

1. **source.db v2→v3**: `migrations/source/003_excision_tombstones.sql` — pure `CREATE TABLE`/`CREATE INDEX IF NOT EXISTS`, one `PRAGMA user_version=3` step, gated by a **verified backup manifest** (durable regime). No change to `raw_sessions` columns → no copy-forward, no consent gate for the schema itself. Update `SOURCE_SCHEMA_VERSION = 3`.
2. **Blob scrubbing is intentional destruction of durable evidence** — that *is* right-to-forget, not an accidental durable mutation. It goes through the normal write path (not a schema migration); the tombstone records `original_blob_hash`/`original_content_hash`/`span_text_hash` so the destruction is auditable (hashes only, never bytes). Must run under a backup manifest + `--yes`.
3. **ops.db**: bump canonical DDL only; `reset --database` recreates it. `mutation_audit` is disposable by design (audit persistence for durable evidence would instead live in source.db if the operator wants it permanent — flag as a design fork, default disposable).
4. **user.db**: no schema step; add enum value + regen openapi/cli-output-schemas + add `SECRET_CANDIDATE` to any `user_audit` every-kind invariant surface (MEMORY.md gotcha).
5. **index/embeddings**: none — derived, rebuilt.
6. New module(s) → regenerate topology projection (`devtools render topology-projection && topology-status`) or `render all --check` fails.

---

## 4. Test strategy

- **Secret round-trip (never-log invariant)** — seed a session with a fake credential (`AKIA…` synthetic). Assert: scanner emits a candidate with correct span coords **and grep the entire emitted flow** (candidate assertion body, logs, `mutation_audit` rows, dry-run output, JSON envelopes) for the credential string → **zero hits**. This is the 27m PITFALL made a gate.
- **Forget-reconciliation invariant** — excise a seeded message, then `grep` across **source raw blob, index rows, FTS, embeddings, blob refs** for the removed text hash → zero; assert tombstone present with both hashes. Then **re-ingest the original source** → assert the secret does not reappear (guard fires; content settles on `excised_content_hash`). This is the 27m primary AC.
- **Reconciliation stage fails loudly** — inject a resurrected row (simulate a rebuild that re-materialized the original) → assert the converger stage raises / writes error `convergence_debt`, does not silently pass.
- **jnj.5 mutation contract** (dedicated tests, both paths): typo'd `--session`/`--source` ref → **zero-target dry-run, zero suppression rows written** (assert `assertions` unchanged); real ref **mutates only with `--yes`**; stable JSON envelope identical shape for dry-run and mutation. Extend the same envelope assertion to `delete_session` (CLI+MCP) — **one shared contract test parametrized across excise/reset/delete/mcp-admin**.
- **Blob fan-out / dedup** — two sessions sharing one blob; excise from one → sibling's blob preserved (ref-count), secret gone from the excised one, present-or-also-excised in sibling per fan-out policy.
- Use `frozen_clock`; seed via `SessionBuilder`; run through `devtools test <files>` (not blanket). Demo-path (`polylogue demo seed`) for the read-boundary masking check — private-data-free.

---

## 5. Bead breakdown (children under `kwsb`; `27m` splits, `jnj.5` folds in)

| # | Bead | Scope | Acceptance |
|---|---|---|---|
| B1 | **Shared mutation contract + `mutation_audit`** | ops.db DDL; `MutationPlan` (resolve→dry-run preview→`--yes`→apply→audit) + `record_mutation_audit`; retrofit `delete_session` (CLI+MCP). | One helper drives reset/excise/delete; stable JSON envelope identical across all; audit row written for every destructive op (dry-run rows have `applied_at_ms=NULL`). |
| B2 | **jnj.5 — route reset `--session`/`--source` through B1** | Fix `reset.py:311-339`; resolve-then-preview-then-`--yes`; typo yields zero-target. | Typo ref → zero-target dry-run, suppression state asserted unchanged; real ref mutates only with `--yes`; `devtools test test_reset` green. Depends B1. |
| B3 | **source.db v3 migration: `excision_tombstones`** | `003_*.sql` + backup-manifest gate + `SOURCE_SCHEMA_VERSION=3` + tombstone read/write helpers. | Migration applies additively under backup manifest; `schema-versioning` lab policy passes; tombstone CRUD tested. |
| B4 | **Excise engine + `ops excise` CLI** | Cross-tier removal (source blob scrub → repoint → drop ref; index session rebuild; FTS; embeddings by ref; tombstone; audit). | Excising a seeded message removes it from every tier (grep across all five) + tombstone with both hashes; blob GC reclaims original. Depends B1,B3. |
| B5 | **Idempotency resurrection guard** | Ingest-time tombstone lookup on `content_hash` → re-excise before write. | Re-ingesting the original source does not resurrect the excised content; content settles on `excised_content_hash`. Depends B3,B4. |
| B6 | **Secret scanner + judgment queue** | Ruleset+entropy detector emitting `SECRET_CANDIDATE` (span coords only); `ops scan-secrets` retro (rate-limited); accept→excise, reject→suppression. | Seeded fake credential flagged **without logging its value**; retro scan produces a bounded candidate list; accept path invokes B4, reject writes suppression. Depends B4. |
| B7 | **Read-boundary overlay + reveal affordance** | Scan-on-read at `get_messages`/read-view; mask by default; gated `--reveal-secrets`/MCP `reveal`/web toggle; CLI+MCP+web. | Read masks candidate spans by default; reveal is explicit + audited; bytes never mutated on read; never-log gate holds. Depends B6. |
| B8 | **Converger reconciliation stage** | `ExcisionReconcileStage`: re-scan index/FTS/embeddings/blob for tombstoned hash/span; fail loudly on reappearance; close `security-privacy-coverage.yaml` gaps. | Injected resurrection → stage raises/alerts, does not pass silently; coverage manifest gaps each have owning bead/test. Depends B4,B5. |

---

## 6. Top-3 risks

1. **Re-acquire ↔ parse-drift asymmetry.** The guard (B5) keys on `original_content_hash`. If parser output drifts (a schema reparse changes the payload), the recomputed hash no longer matches the tombstone → **silent resurrection** of the secret on the next rebuild. Mitigation: tombstone also stores `span_text_hash`; the reconciliation stage (B8) probes tiers by span-text hash independent of session-hash stability, and excision at **block granularity** (block content hash) is more drift-robust than session granularity.

2. **Content-addressed blob dedup / multi-session fan-out.** A secret pasted into N sessions shares one blob (or repeats across blobs). Scrubbing one raw row must (a) not GC-orphan a blob a legitimate sibling still needs — the ref-count in `blob_refs` protects this — and (b) fan out excision to *every* occurrence, or the grep-all-tiers AC fails on a sibling. Mitigation: excise resolves all `blob_refs`/sessions carrying the span before applying; the reconciliation stage is the backstop.

3. **Read-boundary never-log discipline + hot-path cost.** Scan-on-read runs on every `get_messages`; a single logging/telemetry regression that captures the matched span re-leaks what excision removed, and an unbounded regex sweep hurts read latency. Mitigation: the **coordinates-only invariant** enforced by the round-trip grep test across *all* emitted artifacts (B6 test), bounded ruleset + entropy thresholds, and candidate caching keyed on block content hash so unchanged blocks aren't re-scanned.

**No files or beads were modified.** Key paths: `polylogue/core/hashing.py`, `polylogue/pipeline/ids.py`, `polylogue/pipeline/services/ingest_batch/_core.py:394-494`, `polylogue/cli/commands/reset.py:311-339`, `polylogue/storage/sqlite/archive_tiers/source.py`, `polylogue/storage/blob_gc.py`, `polylogue/core/enums.py:399`, `polylogue/mcp/server_mutation_tools.py:753`, `polylogue/api/archive.py:4327`, `docs/plans/security-privacy-coverage.yaml`.

---

## [a41374bad222538dc] Wave-5 Spec: Navigable Knowledge-Graph / "Second Brain" over the archive

I have enough grounding. Here is the tight implementation spec.

---

# Wave-5 Spec: Navigable Knowledge-Graph / "Second Brain" over the archive

**Grounding (real files):** `polylogue/storage/sqlite/archive_tiers/index.py` (`INDEX_SCHEMA_VERSION = 24`; `blocks.search_text` VIRTUAL col at L201-207 = `text ‖ tool_name ‖ $.command ‖ $.file_path ‖ $.path`; `actions` VIEW L324-343; `session_commits` L502-519 with `detection_type/method/confidence/evidence_json`; `web_content_constructs` L244-269 with structural `url`; `session_repos`/`repos`/`session_working_dirs` L470-499; `insight_materialization` L674-687). `polylogue/insights/session_commit.py` already ships `extract_github_refs` (URL, `owner/repo#N`, bare `#N`), `_COMMIT_SHA_RE`, `extract_referenced_files`. `polylogue/core/refs.py` `ObjectRefKind` already has `file/commit/repo/github-issue/github-pr/github-review` (missing `symbol`, `url`). Query grammar (`archive/query/expression.py`) already has a `files` unit source and unit-scoped `… where …` predicates.

**Design stance:** this is a *derived materialization*, not new source truth. It reuses the exact structural/candidate split that `session_commits` already models (`detection_type` + `confidence` + `evidence_json`), and the `insight_materialization` per-session versioned convergence pattern.

---

## 1. Schema / DDL + tier + regime

**Tier: `index.db` (rebuildable, derived).** Everything below is a projection of `blocks`, `actions`, `web_content_constructs`, `session_commits`, `session_repos`. No numbered migration.

**Regime: derived.** Edit canonical DDL in `archive_tiers/index.py`, bump `INDEX_SCHEMA_VERSION 24 → 25`, add a rebuild-plan entry; a schema mismatch triggers blue-green rebuild via `polylogue ops reset --index && polylogued run`. `devtools lab policy schema-versioning` forbids an upgrade helper here. Two touch-points beyond the new tables: extend the `insight_materialization.insight_type` CHECK to add `'entity_mentions'` and `'entity_topics'`, and add `symbol` + `url` to `ObjectRefKind` (`core/refs.py`) + `_OBJECT_REF_KINDS`.

```sql
-- Canonical entity nodes (materialized dedup of all mentions).
CREATE TABLE IF NOT EXISTS entities (
    entity_id     TEXT GENERATED ALWAYS AS (entity_kind || ':' || canonical_key) STORED UNIQUE,
    entity_kind   TEXT NOT NULL CHECK(entity_kind IN
                    ('file','symbol','url','commit','repo','github-issue','github-pr')),
    canonical_key TEXT NOT NULL,          -- normalized: file=abspath|repo-rel; issue=repo#N or _bare#N; url=normalized url
    display_label TEXT NOT NULL,
    repo_id       TEXT REFERENCES repos(repo_id) ON DELETE SET NULL,  -- repo scope for #N disambiguation
    first_seen_ms INTEGER,
    last_seen_ms  INTEGER,
    structural_mentions INTEGER NOT NULL DEFAULT 0 CHECK(structural_mentions >= 0),
    candidate_mentions  INTEGER NOT NULL DEFAULT 0 CHECK(candidate_mentions  >= 0),
    PRIMARY KEY(entity_kind, canonical_key)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_entities_kind_last ON entities(entity_kind, last_seen_ms);
CREATE INDEX IF NOT EXISTS idx_entities_repo ON entities(repo_id) WHERE repo_id IS NOT NULL;

-- Mention edges: entity ← (session,message,block). This is the graph.
CREATE TABLE IF NOT EXISTS entity_mentions (
    mention_id   TEXT GENERATED ALWAYS AS (block_id || ':' || entity_id || ':' || tier) STORED UNIQUE,
    entity_id    TEXT NOT NULL,           -- denormalized (no FK: entities rebuilt in same pass)
    entity_kind  TEXT NOT NULL,
    session_id   TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    message_id   TEXT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
    block_id     TEXT NOT NULL REFERENCES blocks(block_id) ON DELETE CASCADE,
    tier         TEXT NOT NULL CHECK(tier IN ('structural','candidate')),
    source       TEXT NOT NULL CHECK(source IN
                   ('tool_input','web_construct','session_commit','session_repo','prose')),
    confidence   REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    evidence_json TEXT NOT NULL DEFAULT '{}',  -- {raw_match, char_offset, material_origin}
    created_at_ms INTEGER NOT NULL,
    PRIMARY KEY(block_id, entity_id, tier)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON entity_mentions(entity_id, tier, session_id);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_session ON entity_mentions(session_id, entity_kind);

-- Topic clusters (archive-global co-occurrence communities).
CREATE TABLE IF NOT EXISTS entity_topics (
    topic_id     TEXT PRIMARY KEY,        -- stable: hash of sorted seed members
    label        TEXT NOT NULL,           -- top-weighted entity display labels
    method       TEXT NOT NULL,           -- 'label_propagation_v1'
    member_count INTEGER NOT NULL CHECK(member_count >= 0),
    created_at_ms INTEGER NOT NULL
) STRICT;
CREATE TABLE IF NOT EXISTS topic_entities (
    topic_id  TEXT NOT NULL REFERENCES entity_topics(topic_id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL,
    weight    REAL NOT NULL CHECK(weight BETWEEN 0 AND 1),
    PRIMARY KEY(topic_id, entity_id)
) STRICT;

-- Backlinks are a VIEW, not a table (mirrors the `actions` VIEW discipline).
CREATE VIEW IF NOT EXISTS entity_backlinks AS
SELECT em.entity_id, em.entity_kind, em.tier,
       em.session_id, s.title, s.origin, em.message_id, em.block_id,
       em.confidence, em.source
FROM entity_mentions em
JOIN sessions s ON s.session_id = em.session_id;
```

**Why VIEW for backlinks, table for mentions:** mentions are the extracted fact (must persist, indexed both directions); backlinks are a pure join projection — same reason `actions` is a VIEW over paired blocks.

---

## 2. Extraction / graph algorithms (pseudocode)

Per-session materializer `materialize_entity_mentions(session_id, version)` — CPU-bound, runs in `ProcessPoolExecutor`, main process stays sole writer.

```
STRUCTURAL PASS  (tier='structural', confidence 0.9–1.0, provider-reported truth)
  files:
    for row in actions(session_id):                      # actions VIEW
        for p in [row.tool_path] + json_paths(row.tool_input, ['$.file_path','$.path','$.affected_paths[*]']):
            emit(kind='file', key=normalize_path(p, repo_root), source='tool_input', conf=1.0, block=row.tool_use_block_id)
    for wd in session_working_dirs(session_id): register repo_root
  urls:
    for wc in web_content_constructs(session_id) where wc.url is not null:
        emit(kind='url', key=normalize_url(wc.url), source='web_construct', conf=1.0)
        for ref in extract_github_refs(wc.url):           # reuse insights/session_commit.py
            emit(kind='github-'+ref.kind, key=f'{ref.owner}/{ref.repo}#{ref.number}', source='web_construct', conf=1.0)
  commits:
    for sc in session_commits(session_id):                # already structurally scored
        emit(kind='commit', key=sc.commit_sha, source='session_commit', conf=sc.confidence, repo=sc.repo_id)
  repos:
    for sr in session_repos(session_id): emit(kind='repo', key=sr.repo_root, source='session_repo', conf=1.0)

CANDIDATE PASS  (tier='candidate', RECURSIVE-SAFETY GATED, conf 0.3–0.7)
  session_repo := dominant repo_id for session (for #N scoping)
  for block in blocks(session_id) where search_text != '':
      mo := messages[block.message_id].material_origin
      # GATE 1 (authoredness): only human/assistant prose. Excludes runtime_protocol,
      #   runtime_context, generated_context_pack, tool_result — these carry polylogue's
      #   OWN recovery-digest / context-pack citations and must never be re-mined as truth.
      #   (Direct lesson from #2467 recovery-digest fabrication.)
      if mo not in {human_authored, assistant_authored}: continue
      for ref in extract_github_refs(block.text):         # bare #N, owner/repo#N
          key := f'{ref.owner}/{ref.repo}#{ref.number}' if ref.owner
                 else f'{session_repo or "_bare"}#{ref.number}'   # GATE 2: scope bare #N to session repo, never global merge
          emit(kind='github-'+ref.kind, key=key, tier='candidate', source='prose', conf=0.6 if ref.owner else 0.4,
               evidence={raw_match, offset, material_origin:mo})
      for sha in _COMMIT_SHA_RE.finditer(block.text):     # 7–40 hex
          if is_probably_commit(sha):                     # length/entropy heuristic, avoids hex noise
              emit(kind='commit', key=sha.lower(), tier='candidate', source='prose', conf=0.5)
      for sym in SYMBOL_RE.finditer(block.text):          # `Foo.bar`, snake_case fn(), Type::method
          emit(kind='symbol', key=sym, tier='candidate', source='prose', conf=0.3)
      for u in URL_RE.finditer(block.text):
          emit(kind='url', key=normalize_url(u), tier='candidate', source='prose', conf=0.5)

  # GATE 3 (no promotion): candidate rows never rewrite structural rows; PK includes tier
  #   so (block, entity, structural) and (block, entity, candidate) coexist. Readers that
  #   answer "resolve #N to sessions" default to tier='structural'; candidate only on opt-in.

UPSERT entities:  (single global pass after per-session mentions land)
  entities := SELECT entity_kind, canonical_key,
                     min(created)/max(created) as first/last_seen,
                     count(tier='structural') as structural_mentions,
                     count(tier='candidate')  as candidate_mentions
              FROM entity_mentions GROUP BY entity_kind, canonical_key
```

```
TOPIC CLUSTERING  (archive-global, quiet-window batch stage, deterministic)
  # Co-occurrence graph: nodes=entities, edge(a,b) weight = #sessions mentioning both (structural-preferred)
  G := build_cooccurrence(entity_mentions where tier='structural')   # bounded to structural to stay honest
  labels := label_propagation(G, seed=sorted_by_entity_id, iters=fixed)  # deterministic tie-break by entity_id
  for community in group(labels):
      topic_id := sha256(sorted(member entity_ids))[:16]             # stable across rebuilds
      write entity_topics + topic_entities(weight = degree_centrality)
```

**Query surface wiring:** add a `mentions` unit source to the grammar (`archive/query/expression.py`) so `sessions where mentions.entity = "github-issue:polylogue#2467" | count` and `find mentions where entity_kind:file` execute; `resolve_ref` (MCP) gains `entity:` resolution → backlinks. `find every session touching #2467` lowers to a structural-tier `entity_mentions` join.

---

## 3. Rebuild plan

Derived tier → no upgrade helper. Registered as a rebuild plan entry alongside the DDL bump:

1. `INDEX_SCHEMA_VERSION 24 → 25`; schema mismatch on daemon start triggers blue-green index rebuild (existing machinery).
2. New `ConvergenceStage EntityGraphStage` (`daemon/convergence_stages.py`), session-scoped:
   - `check_sessions`: sessions where `insight_materialization('entity_mentions', session_id).materializer_version != CURRENT` (identical pattern to L1435-1466).
   - `execute_sessions`: run `materialize_entity_mentions`, then upsert `entities` for touched keys, stamp `insight_materialization`.
   - Idempotent by `(block_id, entity_id, tier)` PK — re-run is a no-op, matching `session_commits`' `(session_id, commit_sha)` idempotency contract.
3. `EntityTopicStage` — global, `false_means_pending`: does bounded co-occurrence work per invocation and pushes remaining backlog to `convergence_debt` (topics deferred until quiet), same trick as insights deferral.
4. Backfill of the live 38 GB archive: `polylogue ops reset --index && polylogued run` replays the whole tree; entity graph materializes as a convergence stage after ingest+index, no separate migration.

---

## 4. Test strategy

- **Property (Hypothesis, `tests/property`):** extractor idempotence (`materialize` twice ⇒ identical rows); `structural ∩ candidate` never collides on `(block,entity)` beyond the intended dual-tier row; every emitted `canonical_key` round-trips through `ObjectRef.parse/format`; bare-#N keys are always repo-scoped (never global).
- **Recursive-safety regression (protected-class test):** a fixture session whose `material_origin ∈ {runtime_context, generated_context_pack, tool_result}` prose contains `#123` and a fabricated commit SHA ⇒ **zero** candidate mentions. Directly encodes the #2467 recovery-digest fabrication lesson; must never be deleted.
- **Golden extraction:** `SessionBuilder` fixture with known files/URLs/#N/SHA/symbols ⇒ asserted structural vs candidate partition and confidences.
- **Resolve-law:** `find every session touching #2467` (structural) returns exactly the sessions with structural github-issue edges; candidate-only sessions excluded unless `--include-candidates`.
- **Rebuild determinism:** materialize → `ops reset --index` → re-materialize ⇒ byte-identical `entity_topics.topic_id` set (stable hashing) and mention rows.
- **FTS parallel / envelope:** MCP `resolve_entity`/backlinks tool contract + `EXPECTED_TOOL_NAMES` update (discovery test); `render openapi` + `render cli-output-schemas` regenerated for the new `ObjectRefKind` members and unit source.
- **Verification cadence:** `devtools test tests/unit/insights/test_entity_graph.py tests/unit/storage/... -k entity`, `mypy --strict` via `devtools verify`, `render all --check` (grep `out of sync`).

---

## 5. Bead breakdown (6, each with acceptance)

1. **`entity-graph: index.db DDL + schema v25`** — add 4 tables + `entity_backlinks` VIEW; extend `insight_materialization` CHECK; add `symbol`,`url` to `ObjectRefKind`; bump version; rebuild-plan entry. *AC:* `render topology-projection`/`render all --check` green; `polylogue ops reset --index && polylogued run` builds clean on demo seed; `devtools lab policy schema-versioning` passes (no upgrade helper).
2. **`entity-graph: structural extractor`** — files/urls/commits/repos/github-from-url from `actions`, `web_content_constructs`, `session_commits`, `session_repos`, reusing `extract_github_refs`/`extract_referenced_files`. *AC:* golden fixture partitions correct; all rows `tier='structural'` conf ≥ 0.9; idempotent property test green.
3. **`entity-graph: candidate prose miner + recursive-safety gates`** — mine `blocks.search_text` for bare #N / SHA / symbol / URL, gated by `material_origin`, bare-#N repo-scoped, no-promotion. *AC:* recursive-safety regression test green (runtime/context-pack prose ⇒ 0 candidates); candidate rows never overwrite structural.
4. **`entity-graph: EntityGraphStage convergence + entities upsert`** — session-scoped stage with `check_sessions`/`execute_sessions`, `insight_materialization` stamping, entities aggregate. *AC:* re-run no-ops; `convergence_debt` retry path exercised; materializer_version bump re-materializes only stale sessions.
5. **`entity-graph: topic clustering (EntityTopicStage)`** — deterministic label-propagation over structural co-occurrence, `false_means_pending`. *AC:* stable `topic_id` across rebuild; bounded-work backlog defers to debt; membership determinism test green.
6. **`entity-graph: query + MCP surface`** — `mentions`/`entities` unit source in grammar, `entity:` in `resolve_ref`, MCP backlinks/`resolve_entity` tool + contract + `EXPECTED_TOOL_NAMES`, regenerate openapi/cli-output-schemas. *AC:* `find every session touching #2467` resolves via structural edges; discovery + envelope-contract tests green; `render openapi` + `cli-output-schemas` in sync.

---

## 6. Top-3 risks

1. **Bare-`#N` cross-repo/cross-year ambiguity.** `#2467` means polylogue-2467 in a polylogue session, something else elsewhere; naive global merge fabricates false backlinks. *Mitigation:* `canonical_key` embeds session repo scope (`repo#N`, else `_bare#N`); `repo_id` column on `entities`; structural github-from-URL is the only globally-trusted issue edge.
2. **Prose-mining fabrication feedback loop (self-archiving).** Polylogue archives sessions *about polylogue* — recovery digests, context packs, and this very spec cite `#N`/SHAs. Mining those as truth re-manufactures the #2467 fabrication defect at graph scale. *Mitigation:* GATE-1 authoredness (`material_origin` excludes runtime/context-pack/tool_result), GATE-3 no candidate→structural promotion, tier-defaulted reads. This is the single highest-stakes correctness constraint and gets a protected regression test.
3. **Rebuild/materialization cost at 38 GB scale.** Regex over `blocks.search_text` across millions of blocks + O(E) global co-occurrence clustering can dominate convergence. *Mitigation:* lean on `idx_blocks_search_text_populated` partial index; per-session incremental materialization via `insight_materialization` version gate (only stale sessions rescan); CPU stage in `ProcessPoolExecutor`; topic clustering restricted to structural edges and `false_means_pending`-bounded so it never blocks ingest.

**Reused, not reinvented:** `session_commits`' structural/confidence/evidence model, `extract_github_refs`/`_COMMIT_SHA_RE`, the `actions` VIEW, `insight_materialization` versioned convergence, the `files` unit source, and `web_content_constructs` structural URLs. New code is the two extractor passes, the recursive-safety gate, and the topic stage.

---

## [a02ea018bfae5c175] Spec: Semantic Notification / Ambient-Surfacing Policy Layer

I have full grounding. Here is the implementation-grade spec.

---

# Spec: Semantic Notification / Ambient-Surfacing Policy Layer

**Thesis.** The daemon's fan-out (`daemon/notifications.py`), `daemon_events` ledger (`daemon/events.py`), and `HealthAlert` ladder (`daemon/health.py`) already carry *operational* alerts. This layer routes *content* signals through the same fan-out without disturbing the operational path: a new `Notice` (severity + `content` family), standing queries as `daemon_events` producers, a scoring gate for "worth interrupting," durable snooze, and the repeat-mistake nudge. It reuses three existing precedents verbatim: the `AlertDedupState` process-singleton (`convergence_debt_alert.py:86`), the `daemon_events` producer/`query_events_since` ledger, and the hot-file quiet-deferral in `convergence_stages.py`.

**Load-bearing grounding:**
- `send_notifications(alerts, config=cfg.raw)` is the only fan-out entry (`notifications.py:186`); the health loop calls it *only when `overall_status != "ok"`* (`cli.py:710`). Content notices must **not** ride that gate — they get a sibling `route_content_notices()` path.
- `daemon_events(id, ts_ms, kind, operation_id, payload_json)` in **ops.db** (disposable tier), already SSE-streamable via `query_events_since` (`events.py:120`).
- `assertions` in **user.db** (durable): `kind` is plain `TEXT` (no CHECK) → a new `AssertionKind` needs **zero user-tier migration**, only regen of generated surfaces. `confidence REAL` and `context_policy_json` columns already exist (`user.py:25-27`).
- `SUPPRESSION` AssertionKind already exists (`enums.py:412`) with a deterministic id helper (`user_write.py:162`).

---

## (1) Schema + tier placement

Durability axis dictates placement. Nothing here needs a durable **structural** migration.

| Object | Tier | Representation |
|---|---|---|
| **Notification policy** | user.db (durable) | new `AssertionKind.NOTIFICATION_POLICY`; `value_json` = policy doc; `key` = family name |
| **Standing-query notify spec** | user.db (durable) | existing `AssertionKind.SAVED_QUERY` assertion + a `notify` block in `value_json` |
| **Snooze / suppression** | user.db (durable) | existing `AssertionKind.SUPPRESSION`; `scope_ref` = notice anchor-class; `value_json.wake_at_ms` + `value_json.wake_on` predicate |
| **Content notices (ledger)** | ops.db (disposable) | `daemon_events` kind `"notice"` (+ granular `notice.<family>`) |
| **Standing-query cursor** | ops.db (disposable) | `daemon_events` last-seen id per query, or a tiny `notify_cursor` KV in ops (rebuildable) |
| **Token-bucket + dedup state** | in-memory (process singleton) | mirror `AlertDedupState` — resets on restart, matches operational precedent |

**`Notice` model** (extends the fan-out envelope so backends need no changes):

```python
class NoticeFamily(str, Enum):
    OPERATIONAL = "operational"   # existing HealthAlerts, implicit today
    CONTENT     = "content"       # this layer

# HealthSeverity gains: NOTICE = "notice", rank 0 (== OK for operational gate;
# never escalates the operational overall_status). Content routing ignores rank.

class Notice(HealthAlert):          # reuse check_name/tier/message/checked_at
    family: NoticeFamily = NoticeFamily.CONTENT
    anchor: str                      # ObjectRef: session:<id>[:block:<pos>] — citation
    confidence: float                # 0..1, mirrors assertions.confidence
    cite_refs: list[str] = []        # ObjectRefs to evidence (pathology/lesson sessions)
    trigger: str                     # "standing_query" | "repeat_mistake_nudge"
    dedup_key: str                   # (trigger, anchor-class, subject) content hash
```

`build_envelope` (`notification_backends/__init__.py`) gains `family`/`anchor`/`confidence`/`cite_refs` passthrough. Backends stay untouched (they serialize the envelope).

**Notify-policy `value_json` shape** (one assertion per family, `key=family`):

```json
{ "family": "content",
  "cadence": "on-event|daily|weekly",
  "interrupt_threshold": 0.6,
  "bucket": {"capacity": 5, "refill_per_hour": 2},
  "min_confidence": 0.5,
  "quiet_defer": true }
```

**Standing-query notify block** (inside a `SAVED_QUERY` assertion's `value_json`):

```json
{ "query": "sessions where repo:foo and status:abandoned",
  "unit": "sessions",
  "notify_on": "appeared|disappeared|count_crossed",
  "threshold": 3,
  "cadence": "on-event" }
```

---

## (2) "Worth interrupting" scoring + cadence (pseudocode)

```
# Per family, in-memory: TokenBucket{capacity, tokens, refill_per_hour, last_ts}
#                        DedupState (mirror AlertDedupState): dedup_key -> last_emit_ts

def route_content_notices(candidates: list[Notice], now, cfg, snoozes):
    interrupts, ledger_only = [], []
    for n in candidates:
        policy = policy_for(n.family, cfg)          # NOTIFICATION_POLICY assertion

        # -- hard gates (short-circuit) --
        if n.confidence < policy.min_confidence:        ledger_only.append(n); continue
        if is_snoozed(n, snoozes, now):                 ledger_only.append(n); continue
        if is_recursive(n):                             continue          # §recursive-safety, drop entirely
        if policy.quiet_defer and anchor_is_hot(n.anchor):
            defer_until_quiet(n); ledger_only.append(n); continue        # reuse hot-file logic

        # -- score (deterministic, [0,1]) --
        score = ( severity_weight(n)                    # NOTICE=0.4 .. CRITICAL=1.0
                * n.confidence                          # calibrated similarity/evidence
                * recency_factor(n, now)                # newer anchors weigh more
                * novelty_factor(n, dedup, now) )       # decays toward 0 if dedup_key seen recently

        # -- cadence gate --
        if policy.cadence != "on-event":
            ledger_only.append(n); continue             # digest job emits later (§digest)

        # -- fatigue gate: per-family token bucket --
        if score >= policy.interrupt_threshold and bucket(n.family).try_consume(now):
            dedup.mark(n.dedup_key, now)
            interrupts.append(n)
        else:
            ledger_only.append(n)                        # still visible via `brief`

    always_emit_to_ledger(candidates)                    # daemon_events kind="notice" (all of them)
    if interrupts:
        send_notifications(interrupts, config=cfg.raw)   # existing fan-out, unchanged
```

```
def novelty_factor(n, dedup, now):
    last = dedup.last_emit_at.get(n.dedup_key)
    if last is None: return 1.0
    age = now - last
    return min(1.0, age / policy.dedup_window_s)     # linear recovery, mirrors _should_emit

def anchor_is_hot(anchor):
    # reuse convergence_stages hot-file registry: True if the anchor session's
    # source file appended within the quiet window. Defer, don't drop.
```

**Cadence semantics:**
- **on-event** — evaluate + interrupt immediately (subject to bucket + quiet-defer).
- **daily / weekly** — notices land in the ledger only; a `digest` job at the cadence boundary composes **one** roll-up `Notice` per family from `query_events_since` over the window and fans that out. This is `brief` auto-emitted.

**Fatigue control** is the token bucket alone: when a family drains its bucket, excess notices degrade to ledger-only (recoverable via `brief`), never lost. Refill is time-based, so a burst is throttled but a sustained real signal eventually re-fires (matching `_should_emit`'s "at least once per window").

**`polylogue brief --since 24h`** = `query_events_since`/`query_daemon_events(kind="notice")` filtered by window, grouped by family, rendered with anchor + confidence + cite. Pure read over ops.db; no new store.

---

## (3) Migration

The design's sharpest property: **no durable structural migration.**

1. **user.db** — `AssertionKind.NOTIFICATION_POLICY` added to the enum; the `kind` column is `TEXT` with no CHECK (`user.py:12`), so `USER_SCHEMA_VERSION` stays **4**. Snooze reuses `SUPPRESSION`. Existing `idx_assertions_kind_status_updated` already indexes `kind`.
2. **ops.db** — disposable/rebuildable tier, **no migration chain** (per CLAUDE.md schema regimes). New `daemon_events.kind` values (`"notice"`, `notice.<family>`) are additive rows, no DDL. Any `notify_cursor` KV is rebuildable → edit canonical DDL, no upgrade helper.
3. **Generated-surface regen (required, not optional)** — the new AssertionKind is embedded in `render openapi` + `render cli-output-schemas` (CLAUDE.md gotcha). New MCP tools (`brief`, `notify_policy_*`) require `EXPECTED_TOOL_NAMES` + tool-contract updates or discovery tests fail. New `polylogue/` module (e.g. `daemon/content_notices.py`) requires `devtools render topology-projection && topology-status`.

Migration checklist reduces to: enum + `render all`, verify `render all --check` (grep `out of sync`, don't trust the tail line).

---

## (4) Test strategy

Property/contract tests, testmon-affected. Clock via `frozen_clock` (host-clock lint will reject `time.time()` in tests).

1. **Fatigue / token-bucket** — emit N > capacity notices in a burst at frozen T; assert exactly `capacity` fan-out, remainder ledger-only; advance clock by `1/refill_per_hour`; assert one more fires. Mirror `convergence_debt_alert` tests (inject own `TokenBucket`/`DedupState`, no singleton).
2. **Dedup / novelty** — same `dedup_key` twice within `dedup_window_s` → one fan-out, both in ledger; after window → re-fires. Assert `novelty_factor` monotone in age.
3. **No-self-alert (recursive safety)** — feed a candidate whose anchor session has `material_origin ∈ {generated_context_pack, runtime_context, runtime_protocol}` → dropped entirely (not even ledgered as interrupt-eligible). Feed a `daemon_events(kind="notice")` back into the standing-query evaluator → asserts the evaluator's kind-exclusion filters it (no notice-begets-notice loop). Nudge neighbors that include S itself or S's `session_links` lineage → excluded.
4. **Snooze-with-wake** — write a `SUPPRESSION` assertion with `wake_at_ms=T+1h`; assert suppressed at T (ledger-only), fires at T+1h; `wake_on` predicate (e.g. `count_crossed`) fires early when met. Durability: snooze survives `ops reset --index` (it's in user.db) — integration assert.
5. **Cadence** — daily-policy notices never fan out on-event; digest job at boundary emits exactly one roll-up per family; `brief --since` returns the full window regardless of fan-out.
6. **Quiet-defer** — anchor marked hot → deferred; after quiet window → routed. Reuse hot-file test fixtures.
7. **Nudge calibration** — synthetic pathology/lesson session + a near-duplicate live tail → nudge with `confidence` monotone in cosine; below `sim_floor` → no nudge; every emitted Notice carries non-empty `anchor` + ≥1 `cite_refs` (contract invariant, like the existing envelope-contract tests).
8. **Operational isolation** — a `NOTICE`-severity content notice must not raise operational `overall_status` above `ok` (assert `_compute_overall_health` unaffected; the health loop's `!= "ok"` gate never fires on content).

Protected-test parity: add a `test_notice_envelope_contracts.py` in the existing envelope-contract style (`test_envelope_contracts.py`).

---

## (5) Bead breakdown (proposed — do not create; relates to epic `polylogue-9e5`)

`9e5.1` (assertion-adoption audit) is the **gating dependency**: if the assertion flywheel is aspirational, this layer ships dark. Propose a new epic `polylogue-notify` (this is implementation, not the read-only audit lane), `depends_on 9e5.1`.

| # | Title | Acceptance |
|---|---|---|
| notify.1 | `Notice` model + `content` family + envelope passthrough | `Notice` extends `HealthAlert`; `NoticeFamily`/`NOTICE` severity added; `build_envelope` carries `family/anchor/confidence/cite_refs`; content NOTICE never raises operational `overall_status`; contract test green |
| notify.2 | Content ledger + `route_content_notices` + `brief` | notices persist to `daemon_events(kind="notice")`; `route_content_notices` sibling to health loop (does **not** use `!=ok` gate); `polylogue brief --since 24h` reads ledger grouped by family with anchor+confidence; MCP `brief` tool + `EXPECTED_TOOL_NAMES` |
| notify.3 | Scoring gate + per-family token bucket + dedup | deterministic `score∈[0,1]`; `TokenBucket` + `DedupState` process singletons mirroring `AlertDedupState`; over-capacity → ledger-only, recovers on refill; unit tests §4.1–4.2 |
| notify.4 | `NOTIFICATION_POLICY` kind + three cadences + digest job | new AssertionKind (no user migration); on-event/daily/weekly honored; digest job composes one roll-up per family at cadence boundary; `render openapi`+`cli-output-schemas` regenerated; `render all --check` clean |
| notify.5 | Standing-query producers (`notify_on`) | `SAVED_QUERY` assertions with `notify` block evaluated against unit sources; `appeared/disappeared/count_crossed` emit ledger notices; cursor in ops.db; **evaluator excludes `kind LIKE 'notice%'`** (no self-trigger) |
| notify.6 | Durable snooze-with-wake | `SUPPRESSION` assertion `wake_at_ms`+`wake_on`; suppressed→ledger-only until wake; survives `ops reset --index`; CLI `polylogue notify snooze <anchor> --for 24h` |
| notify.7 | Repeat-mistake nudge | embed live tail → `find_similar` vs sessions with `PATHOLOGY`/`LESSON` assertion or in `find_abandoned_sessions`; `confidence` calibrated from cosine; recursive-safety exclusions (self, lineage, generated `material_origin`); quiet-defer on hot anchor; every nudge cites ≥1 evidence ref |
| notify.8 | Recursive-safety + quiet-defer integration | end-to-end: no notice-begets-notice; no alert on generated context; hot-anchor deferral; §4.3/4.6/4.8 pass; `topology-projection` regenerated for new module |

Split as 6–8; notify.1–.3 are the coherent first PR (envelope + ledger + gate), notify.4–.6 second, notify.7–.8 third.

---

## (6) Top-3 risks

1. **Notification fatigue defeats adoption (product risk, highest).** If the token bucket / interrupt threshold are miscalibrated, the operator mutes the whole channel and the layer is dead weight — the exact failure the `9e5.1` adoption audit exists to detect. Mitigation: ship **ledger-first, fan-out-conservative** (default `cadence=daily`, high `interrupt_threshold`, small bucket); `brief` is the primary surface, interrupts are opt-in per family. Gate the whole epic on `9e5.1`.
2. **Recursive amplification (correctness).** A notice written to `daemon_events` re-triggering a standing query, or the nudge embedding a notice-generated context pack, creates a feedback loop that drains buckets and spams. Mitigation is structural and tested (§4.3): evaluator hard-excludes `kind LIKE 'notice%'`; nudge hard-excludes `material_origin ∈ generated/runtime` and the anchor's own `session_links` lineage. This must be an invariant test, not a config knob.
3. **Nudge false-positives erode trust (correctness/calibration).** "You're repeating a past mistake" against a weakly-similar or misattributed pathology session is worse than silence — one bad nudge poisons the channel. Mitigation: hard `min_confidence` floor + `sim_floor` on cosine; mandatory citation-anchor + `cite_refs` so every nudge is auditable to its evidence; "ended badly" must be an explicit `PATHOLOGY`/`LESSON`/abandoned signal, never prose-mined (consistent with the recovery-digest honesty fix in project memory — text-mined events became *candidates*, not claims).

---

**Grounding files:** `polylogue/daemon/notifications.py` (fan-out), `daemon/events.py` (ledger), `daemon/health.py` (HealthAlert/severity/tiers), `daemon/convergence_debt_alert.py` (dedup-state precedent to mirror), `daemon/cli.py:697-713` (health loop — sibling routing point), `storage/sqlite/archive_tiers/user.py:12` (assertions DDL — no migration needed), `core/enums.py:399` (AssertionKind). No code or beads were modified.

---

## [a86c12a679c6bb13d] Wave-5 Spec — Session Replay / Reproduction

I have enough grounding. Here is the spec.

---

# Wave-5 Spec — Session Replay / Reproduction

Grounded in: `actions` VIEW (`storage/sqlite/archive_tiers/index.py:324`), `session_commits` (`:502`), `session_runs.cwd`/`git_branch` (`:931`), `session_agent_policies` (approval/sandbox/network, `:360`), `session_events` (`:345`), and the existing DERIVED-model replayer `maintenance/replay.py`.

## 0. Naming caveat (load-bearing)
`polylogue/maintenance/replay.py::execute_replay` already exists — but it replays **derived-model repair targets** (FTS/insights backfill), not sessions. Do **not** overload it. New surface lives under `polylogue/reproduction/` (or `insights/replay/`). Reuse only its *patterns*: opaque cursor, JSON state file under archive root, `false_means_pending`/resume, bounded `FailureSample`.

---

## 1. What the archive already gives us (the substrate)

The ordered tool-call stream is **already queryable** — no new extraction needed for phase 1:

```sql
-- Ordered executable step list for one session
SELECT m.position AS msg_pos, b.position AS blk_pos,
       a.tool_name, a.tool_command, a.tool_path, a.tool_input,
       a.is_error, a.exit_code
FROM actions a
JOIN blocks b   ON b.block_id = a.tool_use_block_id
JOIN messages m ON m.message_id = a.message_id
WHERE a.session_id = ?
ORDER BY m.position, b.position;
```

- **cwd + git branch** come from `session_runs.cwd` / `git_branch` (per-run, so a resumed/forked session has multiple).
- **recorded git SHA** comes from `session_commits` — but only `detection_type='explicit_ref'` (conf 0.95) or `'origin_reported'` are trustworthy anchors; `time_window`/`file_overlap` are heuristic and MUST NOT gate a re-execution checkout.
- **provider outcome** (`is_error`, `exit_code`) is the recorded oracle to diff against on re-run (index v16 keystone) — `NULL`=unknown, never inferred.
- **capture integrity** from `session_events` (`compaction`, `capture_gap`) — a `capture_gap` between two steps means the step list is provably incomplete → not replayable.

---

## 2. Schema — replay plan / result + tier

**Tier: `ops.db` (disposable).** The DRY plan is a pure projection of `index.db` (rebuildable → never durable). Execution results and RL rewards are *expensive but reproducible* evaluation output → default to `ops.db` bookkeeping, with an explicit export path (file artifact under `.local/reproduction/` or a `user.db` assertion `kind=judgment`) when an operator wants to keep a reward corpus. This respects the durability axis: nothing here is irreplaceable-by-construction.

```sql
-- ops.db (bump ops user_version; disposable-tier rebuild, not a numbered migration)

CREATE TABLE IF NOT EXISTS replay_runs (
    replay_id       TEXT PRIMARY KEY,          -- uuid4
    session_id      TEXT NOT NULL,             -- logical session replayed
    mode            TEXT NOT NULL CHECK(mode IN ('dry','execute')),
    anchor_sha      TEXT,                       -- checkout SHA (NULL for dry / no anchor)
    anchor_source   TEXT CHECK(anchor_source IN ('explicit_ref','origin_reported','operator','none')),
    cwd             TEXT,                        -- resolved from session_runs.cwd
    repo_id         TEXT,
    step_count      INTEGER NOT NULL,
    replayable      INTEGER NOT NULL CHECK(replayable IN (0,1)),
    skip_reason     TEXT,                        -- why not replayable / not executed
    determinism     REAL,                        -- fraction of steps matching recorded outcome (execute only)
    status          TEXT NOT NULL CHECK(status IN ('planned','running','completed','failed','aborted')),
    cursor          TEXT,                        -- resume: 'step:N' (mirrors maintenance/replay cursor)
    created_at_ms   INTEGER NOT NULL,
    completed_at_ms INTEGER
) STRICT;

CREATE TABLE IF NOT EXISTS replay_steps (
    replay_id       TEXT NOT NULL REFERENCES replay_runs(replay_id) ON DELETE CASCADE,
    step_index      INTEGER NOT NULL,            -- 0-based, msg_pos/blk_pos order
    tool_use_block_id TEXT NOT NULL,             -- citation anchor back into blocks
    tool_name       TEXT NOT NULL,
    classification  TEXT NOT NULL CHECK(classification IN
                       ('pure_read','deterministic_write','side_effecting','network','nondeterministic','unknown')),
    recorded_is_error   INTEGER,                 -- oracle from actions.is_error
    recorded_exit_code  INTEGER,
    replay_is_error     INTEGER,                 -- execute mode
    replay_exit_code    INTEGER,
    outcome_match   TEXT CHECK(outcome_match IN ('match','regressed','newly_passes','divergent','skipped','unknown')),
    reward          REAL,                         -- phase 2 RL signal, per-step
    detail          TEXT,
    PRIMARY KEY(replay_id, step_index)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_replay_steps_class ON replay_steps(replay_id, classification);
```

In-memory domain types (Pydantic/dataclass, `reproduction/models.py`): `ReplayPlan(session_id, anchor, cwd, steps: tuple[ReplayStep,...], replayable, skip_reason)` and `ReplayStep(index, block_id, tool_name, classification, recorded_outcome)`. The plan is **printable without touching ops.db** — persistence is only for `execute` mode.

---

## 3. Replay algorithm + safety carve-outs (pseudocode)

```
def build_plan(session_id) -> ReplayPlan:            # PHASE 1 — pure, no execution
    runs   = session_runs(session_id)                # cwd, git_branch per run
    steps  = query_actions_ordered(session_id)       # the SQL in §1
    anchor = pick_anchor(session_commits(session_id)) # only explicit_ref/origin_reported

    replayable, reason = classify_session(session_id, runs, steps)
    plan_steps = [ReplayStep(i, s.block_id, s.tool_name,
                             classification=classify_tool(s.tool_name, s.tool_input),
                             recorded=(s.is_error, s.exit_code))
                  for i, s in enumerate(steps)]
    return ReplayPlan(session_id, anchor, cwd=runs.cwd, steps=plan_steps,
                      replayable=replayable, skip_reason=reason)

def classify_session(session_id, runs, steps) -> (bool, reason):
    if session_events(session_id).has('capture_gap'):   return (False, 'capture_gap: step list provably incomplete')
    if origin(session_id) not in {codex-session, claude-code-session}:
                                                        return (False, 'non-agentic origin: no executable tool stream')
    if runs.cwd is None:                                return (False, 'no recorded cwd')
    if any(s.classification == 'network' for s in steps):
                                                        return (False, 'network side effects — not reproducible')
    return (True, None)

def classify_tool(name, input) -> classification:
    match name:
      Read|Grep|Glob|LS                         -> pure_read
      Bash where cmd ∈ SAFE_READONLY_ALLOWLIST  -> pure_read      # git status, ls, cat, rg…
      Edit|Write|NotebookEdit                   -> deterministic_write
      Bash (mutating, cwd-scoped)               -> side_effecting
      WebFetch|WebSearch|curl|git push|gh|network-> network
      Task (subagent)|time/random-bearing       -> nondeterministic
      _                                         -> unknown

# PHASE 2 — GATED execution (default OFF; requires --execute and an anchor)
def execute_plan(plan) -> ReplayRun:
    assert plan.replayable and plan.anchor and operator_opt_in
    sandbox = fresh_worktree(plan.repo_id, checkout=plan.anchor.sha)   # ephemeral, /realm/tmp/worktrees
    guard   = SandboxGuard(root=sandbox, deny_network=True, deny_paths_outside_root=True,
                           frozen_clock=plan.anchor.commit_time, seeded_random=True)
    run = new_replay_run(status='running', cursor=resume_or('step:0'))
    for step in plan.steps[resume_index:]:
        if step.classification in {network, nondeterministic}:  record(step, 'skipped'); continue
        if step.classification == unknown:                      record(step, 'skipped'); continue   # fail-closed
        result = guard.exec(step)                               # side effects confined to worktree
        step.outcome_match = diff(result, step.recorded_outcome)  # match/regressed/newly_passes/divergent
        step.reward = reward_fn(step.outcome_match)
        run.cursor = f'step:{step.index+1}'; checkpoint(run)    # resume like maintenance/replay
    run.determinism = fraction(match) ; destroy(sandbox); return run
```

**Safety carve-outs (fail-closed):**
1. **Network deny by default** — any `network` step aborts planning-as-replayable; in execute mode the guard blocks egress (no `gh`/`curl`/`git push`/`WebFetch`).
2. **Filesystem containment** — execution ONLY inside a throwaway git worktree at the anchor SHA under `/realm/tmp/worktrees/`; writes outside the worktree root are denied. Never touches the operator's live checkout or `/realm/data`.
3. **`unknown`/`nondeterministic` → skipped, never guessed** — mirrors the `NULL`=unknown discipline of the tool-result keystone.
4. **Time/entropy** — `frozen_clock` at the commit time + seeded RNG (reuse `tests/infra/frozen_clock.py`).
5. **Respect recorded sandbox policy** — if `session_agent_policies.sandbox_policy`/`network_policy` for the run were more restrictive than the replay guard, the guard takes the intersection (never looser than the original).
6. **Two-phase gate** — `execute` requires all of: `replayable=1`, a trustworthy anchor, operator `--execute`, and a clean ephemeral sandbox. Any missing → dry only.

---

## 4. Migration

`ops.db` is the **disposable** tier: no numbered migration chain (that regime is source/user only). Per the schema-regime rule enforced by `devtools lab policy schema-versioning`:
- Add the two tables to the canonical `ops.db` DDL, bump its `PRAGMA user_version` (currently 1 → 2), and record the rebuild path (`polylogue ops reset --ops && polylogued run`), not an upgrade helper.
- Because ops.db is disposable, the "migration" is a reset — no backup manifest required (that's a durable-tier obligation).
- If a durable reward corpus is later wanted, that is a **separate** `user.db` numbered additive migration behind a backup manifest — explicitly out of scope for the first slices.

---

## 5. Test strategy

- **Plan projection (property/unit):** for a `SessionBuilder` session with N tool_use/tool_result pairs, `build_plan` yields exactly N ordered steps with correct `(msg_pos, blk_pos)` order and recorded outcomes matching the `actions` VIEW. Use `corpus_seeded_db`.
- **Classification table (unit, parametrized):** each tool name/input → expected classification; assert Bash allowlist vs mutating split; assert `unknown` is the default (fail-closed).
- **Replayability gate (unit):** sessions with `capture_gap`, missing cwd, network step, or non-agentic origin all classify `replayable=0` with the right `skip_reason`.
- **Anchor selection (unit):** only `explicit_ref`/`origin_reported` rows in `session_commits` become anchors; `file_overlap`/`time_window` never do.
- **Dry determinism (integration, protected dir):** a fixture repo + synthetic Codex session; `execute` in a temp worktree; assert `deterministic_write`/`pure_read` steps report `match`, injected regressions report `regressed`, network steps `skipped`. Never run against real `~/.codex` corpus (fixtures only, per cloud-lane rules).
- **Sandbox containment (security, protected):** attempt a write outside the worktree root and a network call → both denied; add under `tests/unit/security/`.
- **Resume (unit):** kill mid-run, re-invoke with same `replay_id`, assert cursor resumes at first unprocessed step, no duplicate `replay_steps`.
- Verify via `devtools test tests/unit/reproduction` + the one integration file — not blanket runs.

---

## 6. Bead breakdown (start with a DRY replayer)

1. **`repro-1` — Ordered action-stream projection.** Read-only `reproduction/plan.py::build_plan` + `ReplayPlan`/`ReplayStep` models, backed by the §1 SQL. **AC:** given a seeded multi-tool session, returns ordered steps with recorded `is_error`/`exit_code`; no ops.db writes; unit + property tests green.
2. **`repro-2` — Tool classification + replayability gate.** `classify_tool` table + `classify_session` (capture_gap / cwd / network / origin). **AC:** parametrized classification tests; all four skip reasons covered; `unknown` is default.
3. **`repro-3` — Anchor resolution from `session_commits` + `session_runs`.** **AC:** only `explicit_ref`/`origin_reported` yield anchors; cwd/branch resolved per run; heuristic detections rejected with reason.
4. **`repro-4` — DRY `polylogue reproduce SESSION --plan` CLI/MCP surface.** Prints executable plan (steps, classifications, anchor, skip_reason) as text + JSON; no execution. **AC:** query-first CLI verb wired (params last), JSON schema regenerated (`render cli-output-schemas`), demo-seed session renders a plan.
5. **`repro-5` — ops.db `replay_runs`/`replay_steps` schema + user_version bump + reset plan.** **AC:** DDL added, `devtools lab policy schema-versioning` passes, `ops reset --ops` rebuilds cleanly, no numbered-migration added.
6. **`repro-6` — SandboxGuard (worktree + network deny + fs containment + frozen clock/seeded RNG).** No step execution yet — just the confined harness. **AC:** security tests: egress blocked, out-of-root write blocked, clock/RNG frozen.
7. **`repro-7` — Gated `--execute` engine + resume cursor + determinism scoring.** Executes only `pure_read`/`deterministic_write`/`side_effecting` steps in the sandbox; diffs against recorded outcome; resumable. **AC:** integration determinism test (match/regressed/skipped) + resume test; `execute` refused without anchor+opt-in.
8. **`repro-8` — Per-step reward emission + export path.** `reward_fn(outcome_match)` → `replay_steps.reward`; optional export to `.local/reproduction/` artifact or `user.db` assertion. **AC:** reward populated for executed steps; export round-trips; worked-then-breaks (`regressed`) surfaces as negative reward.

Land 1–4 first (fully DRY, zero execution risk) — that alone delivers "print an executable plan + catch missing anchors." 5–8 are the gated re-execution phase.

---

## 7. Top-3 risks

1. **False anchors → replaying against the wrong tree.** `session_commits` is mostly heuristic (`time_window`/`file_overlap`, conf ≤0.3). If execute ever accepts those, you re-run at a divergent SHA and mislabel real code as "regressed." Mitigation: hard-gate execute on `explicit_ref`/`origin_reported` only; make anchor provenance a first-class, tested predicate (bead `repro-3`).
2. **Silent step incompleteness.** Auto-compaction and `capture_gap` mean the archived tool stream can be a *subset* of what actually ran (lineage replays only the divergent tail; parents may be un-ingested). Replaying a partial stream produces confident-but-wrong determinism numbers. Mitigation: refuse replay on any `capture_gap`; for forked/resumed sessions, recompose the full logical stream (parent-prefix + child-tail) before planning, or refuse if the parent isn't ingested.
3. **Side-effect escape / non-reproducibility masquerading as regression.** Bash steps can mutate outside cwd, hit the network, or depend on wall-clock/absolute paths/installed toolchain versions absent at replay time — producing `divergent` that looks like a real regression. Mitigation: fail-closed classification (`unknown`→skip), strict worktree+network containment, frozen clock/seeded RNG, and record toolchain/env in the run so `divergent` can be attributed to environment drift rather than model change before any RL reward is trusted.

---

## [aa1dd7e707fb984f4] Spec: Blob-GC correctness hardening — property-tested two-invariant state machine

I have a complete, grounded picture. Here's the tight spec.

---

# Spec: Blob-GC correctness hardening — property-tested two-invariant state machine

**Bead:** `polylogue-8jg9.2` (P2, area:storage/ops, parent `polylogue-8jg9` operational-resilience epic). The prompt's "#818" maps to the `test-closure-matrix.yaml` *known_gap* at `storage.blob_store` (line 180): *"blob GC lease/orphan audit (#818, #1000) not exercised by a dedicated test."* #818/#1000 are legacy GitHub issue numbers preserved in that gap note; the live tracking unit is the bead.

## 0. Ground truth (what already exists)

- `polylogue/storage/blob_gc.py` — full impl: `run_blob_gc`/`run_blob_gc_report`, `acquire_blob_leases`, `release_operation_leases`, `sweep_orphaned_blob_leases`, `read_gc_history`; three documented safety invariants + a fourth generation-age gate.
- DDL is **inline** in `polylogue/storage/sqlite/archive_tiers/source.py` (durable `source.db`, `SOURCE_SCHEMA_VERSION = 2`): `blob_refs`, `pending_blob_refs`, `gc_generations` — there is **no `.sql` migration file**.
- Lease integration: `polylogue/archive/write_effects.py:commit_archive_write_effects` acquires before the data commit (separate immediate-commit conn), releases on **every** exit path (#1746), `_release_leases_on_failure` backstop.
- Startup sweep: `polylogue/daemon/cli.py:_sweep_orphaned_blob_leases_sync`.
- Tests already present and non-trivial: `tests/unit/storage/test_blob_gc_concurrency.py` (threaded interleaving + a small `@given` shuffle), `test_blob_gc_lease_recovery.py` (failure-path lease release, sweep bounds, generation gate, combined-guards), `tests/property/test_blob_store_props.py` (orphan-count/bytes/sample bounds).

**So the bead's literal AC is ~80% met.** The Wave-5 mandate is the *harder* thing the closure matrix still lacks: a `RuleBasedStateMachine` that models the acquire→commit interleaving as a nondeterministic transition system and asserts the two global invariants, plus SIGKILL fault injection and reconciliation of the **two divergent orphan surfaces**.

---

## 1. Schema / state model

Two persistent relations + on-disk set, all in durable `source.db`:

```
pending_blob_refs(blob_hash BLOB[32], operation_id, ref_type, ref_id, acquired_at_ms)   -- the LEASE set L
blob_refs        (blob_hash BLOB[32], ref_id, ref_type, source_path, size_bytes, acquired_at_ms) -- durable ref set R_blob
raw_sessions     (raw_id=hex hash, blob_hash BLOB[32], …)                                -- durable ref set R_raw
gc_generations   (generation_id, started_at_ms, completed_at_ms NULLABLE, reclaimed_count, reclaimed_bytes) -- generation ledger G
DISK             = { hash : file at {root}/{hash[:2]}/{hash[2:]} , mtime }               -- physical set D
```

Abstract model (what the state machine tracks):

```
Reachable(h)  := h ∈ R_raw ∪ R_blob ∪ raw_sessions.raw_id       -- union computed by _reference_surfaces
Leased(h)     := ∃ row in pending_blob_refs with blob_hash=h     -- _has_active_lease
Age(h)        := now - D[h].mtime
Eligible(h)   := h ∈ D ∧ ¬Reachable(h) ∧ ¬Leased(h)
                 ∧ Age(h) ≥ max(MIN_AGE_S, now - last_completed_generation.completed_at)
```

**Two global invariants the machine proves:**

- **I1 (no false reclaim / safety):** at no observable point does GC unlink `h` while `Reachable(h) ∨ Leased(h)` holds *at the instant the unlink commits*. Contrapositive of the whole point.
- **I2 (bounded liveness / no permanent leak):** any `h` that becomes `¬Reachable ∧ ¬Leased` and stays so is physically gone **within two completed GC generations** (one to advance the age high-water mark past the blob, one to reclaim). "No leaked blob survives two GC generations."

Model state carried by the RSM: `blobs: dict[hash → BlobModel(on_disk, reachable, leased_by:set[op], mtime)]`, `generations: list[completed_at]`, `now: FrozenClock`, plus a shadow `ever_reachable_while_present` audit log for I1.

---

## 2. GC / lease state machine (pseudocode)

The real lifecycle the tests must model — five events interleave freely:

```
# ---- writer op lifecycle (acquire → commit → release) ----
ACQUIRE(op, H):                      # write_effects, separate immediate-commit conn
    for h in H: INSERT OR IGNORE pending_blob_refs(h, op, now_ms); commit   # lease visible NOW
COMMIT(op, H):                       # main data txn
    for h in H: INSERT raw_sessions/blob_refs(h); commit                    # h now Reachable
RELEASE(op):                         # post-commit, same conn
    DELETE pending_blob_refs WHERE operation_id=op; commit
ABORT(op):                           # failure path
    (no raw row inserted); _release_leases_on_failure(op)                   # lease dropped, blob stays orphan

# ---- GC pass (the reader/deleter) ----
RUN_GC(max_batch):
    prev = last generation with completed_at NOT NULL
    older_than = max(MIN_AGE_S, now - prev.completed_at)      # inv #3 age gate
    cands = [h in DISK if now - mtime(h) >= older_than] sorted by mtime
    started = now
    for h in cands (bounded by max_batch, EACH check on the live latest-committed snapshot):
        if reference_surfaces(h): skip_referenced; continue    # inv #1  — re-read, not cached
        if has_active_lease(h):   skip_leased;     continue    # inv #2  — re-read, not cached
        stat size; unlink(sharded_path(h))                     # FileNotFoundError→skip_missing
        deleted++; reclaimed += size
    INSERT gc_generations(started, completed_at=now, deleted, reclaimed); commit   # advances G

# ---- fault + recovery ----
SIGKILL(op): drop op's in-flight COMMIT/RELEASE; pending_blob_refs row for op SURVIVES (leaked lease)
SWEEP(max_age): DELETE pending_blob_refs WHERE acquired_at_ms < now-max_age; commit   # daemon startup
CLOCK_ADVANCE(dt): now += dt
```

**Load-bearing correctness fact the model must encode (and the test must not paper over):** each per-candidate `reference_surfaces`/`has_active_lease` call in `RUN_GC` executes as a **standalone `SELECT` in Python-sqlite3 autocommit** (isolation only wraps DML), so it observes the *latest committed* leases/refs — not a snapshot frozen at pass start. I1 holds **only** because of this. A regression that wraps the GC loop in an explicit transaction (freezing the read snapshot before `ACQUIRE` commits) silently breaks I1. The RSM's interleaving of `ACQUIRE` between `RUN_GC` candidate-selection and per-candidate-check is the exact adversary that catches it.

---

## 3. Migration

**Path A — test-only hardening (the bead's literal scope): NO migration.** `blob_refs`/`pending_blob_refs`/`gc_generations` are unchanged; deliver tests + the orphan-surface reconciliation (§5 B4, code-only, no columns).

**Path B — if the sweep-vs-slow-writer race (Risk R2) is closed with a lease heartbeat:** that is one **additive durable** migration on `source.db`, `SOURCE_SCHEMA_VERSION 2 → 3`, per the durable-tier regime (`devtools lab policy schema-versioning`):

```
storage/sqlite/migrations/source/003_pending_blob_refs_heartbeat.sql
  ALTER TABLE pending_blob_refs ADD COLUMN heartbeat_at_ms INTEGER;   -- nullable, additive only
  -- backfill: existing rows keep NULL → sweep falls back to acquired_at_ms
```
Plus: bump `SOURCE_SCHEMA_VERSION`, one `PRAGMA user_version` step, backup-manifest gate, `sweep` reads `COALESCE(heartbeat_at_ms, acquired_at_ms)`, long ops call a new `refresh_blob_leases(op)`. **Recommend deferring Path B to its own bead** — keep this deliverable additive-derived/test-only so it does not drag a durable-tier bump + backup drill onto a test PR.

---

## 4. Property / fuzz strategy — `RuleBasedStateMachine`

New file: `tests/property/test_blob_gc_race_machine.py` (property tier; Hypothesis `stateful`). One real `BlobStore` + one bootstrapped split-file archive per run (`initialize_active_archive_root`), `FrozenClock` for all time (satisfies `verify-test-clock-hygiene`; never `time.sleep`/wall-clock — advance the clock as a rule).

```python
class BlobGCRaceMachine(RuleBasedStateMachine):
    ops     = Bundle("ops")
    blobs   = Bundle("blobs")

    @rule(target=blobs, payload=st.binary(min_size=1, max_size=64))
    def stage_blob(payload):        # write_from_bytes; model.on_disk=True, reachable=False, leased=∅
    @rule(target=ops, h=blobs)
    def acquire(h):                 # acquire_blob_leases; model.leased[h] |= {op}
    @rule(op=ops, h=blobs)
    def commit(op, h):              # insert raw_sessions row; model.reachable[h]=True
    @rule(op=ops)
    def release(op):                # release_operation_leases; drop op from all model.leased
    @rule(op=ops)
    def abort(op):                  # _release_leases_on_failure; op dropped, blob stays orphan
    @rule(op=ops)
    def sigkill(op):                # DELETE the op token WITHOUT touching pending_blob_refs → leaked lease
    @rule(max_age=st.sampled_from([0, ORPHAN_LEASE_MAX_AGE_S]))
    def sweep(max_age):             # sweep_orphaned_blob_leases; model drops leases older than max_age
    @rule(dt=st.integers(0, 5000))
    def advance_clock(dt):          # FrozenClock += dt (drives age gate + generation boundary)
    @rule()
    def run_gc():                   # run_blob_gc_report; RECONCILE model vs disk

    @invariant()
    def i1_no_reachable_or_leased_ever_deleted():
        # for every h the model says reachable OR leased → BlobStore.exists(h) is True
    @invariant()
    def i2_orphan_gone_within_two_generations():
        # any h that has been ¬reachable ∧ ¬leased across ≥2 completed generations → not exists(h)
    @invariant()
    def disk_matches_model_for_settled_blobs():
        # no phantom deletes/survivors among blobs with no in-flight op
```

Config: `settings(max_examples≈150, stateful_step_count≈40, deadline=None, suppress_health_check=[function_scoped_fixture, too_slow])`; register `HYPOTHESIS_PROFILE=ci` cap for cloud lane.

**Convergent-design meta-property:** because the model is provider/shape-agnostic and reconciles *only* against `BlobStore.exists` + the three tables, the same machine transparently exercises any future ref surface added to `_reference_surfaces` — no per-surface test. The machine *is* the coverage.

**Fault-injection lane** (separate, deterministic — not stateful): SIGKILL-mid-transaction realism via a subprocess that `acquire→(os._exit(9) before release)`, parent asserts the lease leaks, then `sweep_orphaned_blob_leases` past `ORPHAN_LEASE_MAX_AGE_S` reclaims it and GC then collects the blob. This gives a *real* killed-process lease (the in-process `sigkill` rule only simulates the leak).

---

## 5. Bead breakdown (children of `polylogue-8jg9.2`; do not create — proposed)

| # | Title | Acceptance |
|---|---|---|
| **B1** | RSM over acquire→commit→GC race (`test_blob_gc_race_machine.py`) | Stateful machine with the 9 rules + I1/I2 invariants runs ≥150 examples green; a deliberate mutant (wrap GC loop in one txn / drop the lease check) makes it fail deterministically. `devtools test tests/property/test_blob_gc_race_machine.py`. |
| **B2** | Real-SIGKILL fault-injection lane | Subprocess killed between acquire and release leaves exactly one `pending_blob_refs` row; `sweep_orphaned_blob_leases` reclaims it only past `ORPHAN_LEASE_MAX_AGE_S`; post-sweep GC collects the now-orphan blob. Clock-frozen, no `sleep`. |
| **B3** | I2 liveness / two-generation reclaim proof | Dedicated test: an orphan present at generation N is gone by end of generation N+2; a blob that *re-acquires* a lease before the second pass survives. Encodes the generation-age-gate math against `FrozenClock`. |
| **B4** | Reconcile the two orphan surfaces (**real bug**) | `BlobStore.detect_orphans/cleanup_orphans` (used by `ops doctor`) is made lease-aware **and** generation-aware, or documented as advisory-only and hard-gated behind `run_blob_gc`; test proves `cleanup_orphans(dry_run=False)` can no longer delete a blob with an active `pending_blob_refs` lease. (See R1.) |
| **B5** | Close the closure-matrix gap | `test-closure-matrix.yaml` `storage.blob_store` known_gap for #818/#1000 removed; `representative_tests` lists the new files; `devtools render all --check` green (grep for `out of sync`, don't trust tail). |
| **B6** *(opt, own PR)* | Lease heartbeat vs slow-writer sweep (Path B) | source.db 2→3 additive migration + `refresh_blob_leases`; sweep uses `COALESCE(heartbeat_at_ms, acquired_at_ms)`; a >`ORPHAN_LEASE_MAX_AGE_S` op that heartbeats is not swept. Behind backup-manifest gate. |

Do B1–B5 as **one coherent PR** (test + the B4 code fix + matrix update); B6 separate because it touches a durable tier.

---

## 6. Top-3 risks

1. **R1 — divergent orphan surfaces (a genuine live bug).** `run_blob_gc` is lease/ref/generation-safe, but `BlobStore.detect_orphans`/`cleanup_orphans` compares disk against **only `raw_sessions.raw_id`** and consults **neither `pending_blob_refs` nor `blob_refs` nor the generation gate**. `polylogue ops doctor` → `cleanup_orphans(dry_run=False)` can therefore delete an **in-flight leased** blob or a `blob_refs`-only attachment. This is exactly the "#818 real orphan-detection bugs" the bead cites. The safety of the whole subsystem is only as good as its *weakest* deletion path. → **B4 is not optional.**

2. **R2 — sweep vs. legitimately slow writer.** `sweep_orphaned_blob_leases` reclaims any lease older than `ORPHAN_LEASE_MAX_AGE_S=3600` by wall-age with **no liveness signal**. A real ingest that legitimately holds a lease >1h (huge Codex/Claude-Code JSONL, backpressure) gets its lease swept, after which a concurrent GC can delete the blob it is mid-committing → I1 violation with zero error surfaced. Bounded today only by "ingests are usually fast." Path B (heartbeat) is the durable fix; until then the RSM should include a `sweep(max_age=0)` adversary to document the exposure.

3. **R3 — the invariant rests on undocumented sqlite3 autocommit-per-SELECT semantics.** I1 holds because each per-candidate check re-reads the latest committed state (Python sqlite3 wraps only DML in implicit transactions). This is load-bearing and invisible: a well-meaning refactor that opens an explicit read transaction around the GC candidate loop (for "consistency" or speed) freezes the snapshot before concurrent `ACQUIRE` commits and silently reintroduces the acquire→commit race. Mitigation: the RSM mutant test in B1 must fail on exactly that transformation, and a comment in `run_blob_gc` must state the autocommit dependency explicitly.

**Minor:** the `#1190` note in `test_blob_gc_concurrency.py:test_release_followed_by_gc_collects_orphan` is stale — `run_blob_gc` now unlinks via `_sharded_blob_path`, so the file *is* removed; the test comment claiming "file is not actually removed" should be corrected when B1–B5 land.

---

## [a2912301b0902d3b8] Lane 6 — Archive Economics & Longevity (10yr / 500GB horizon)

Enough grounding. Here's the lane output.

---

## Lane 6 — Archive Economics & Longevity (10yr / 500GB horizon)

Current substrate facts anchoring these: content-addressed SHA-256 blob store (uncompressed JSON, ~36GB), lease+generation-safe GC (`blob_gc.py`), superseded-snapshot retention (`raw_retention.py`, keeps N recent per source_path/index), lineage stores only divergent tails, 5 tiers split by durability (source/user = durable, index/embeddings = rebuildable, ops = disposable), 13.6GB referenced-but-unfetched attachment bytes, 39,586 missing referenced blobs.

- **Storage-growth ledger as a first-class insight (`bytes/session/day` by tier × origin, with a fitted trend + 1/3/10yr projection)** — you cannot manage affordability you don't measure; today growth is inferred by eyeballing `du`. Make it a materialized insight over `raw_sessions.blob_size` + tier file sizes + ingest cursor timestamps so "when do we hit 500GB?" is one query, and regressions (a provider suddenly 10×-ing bytes/session) alarm early. — NEW (child of polylogue-83u sibling: "affordability telemetry")

- **Ship zstd blob compression now, but decouple it from the attachment epic** — 83u.5 is P3 buried under a P1 attachment program, yet it's the single biggest, lowest-risk affordability win (36GB→~4-7GB, backup surface shrinks proportionally, address stays SHA-256-of-uncompressed so zero schema change). Self-identifying zstd magic means read-path sniff is safe. Promote to its own P2; it pays for itself independent of whether attachment capture ever lands. — polylogue-83u.5 (re-parent / re-prioritize)

- **Compression dictionary trained per-origin** — raw provider JSON is extremely repetitive within an origin (same envelope keys, tool-schema boilerplate, system-prompt prefixes). A zstd trained dictionary (`--train` over a sample per origin) on write-time level-3 frames pushes 5-10× toward the high end for small blobs where generic zstd underperforms. Store dict id in a sidecar, not the address. — NEW

- **Tiering by access-temperature, not just durability** — the 5 tiers are keyed on *durability* (rebuildable vs irreplaceable) but affordability at 10yr needs an orthogonal *heat* axis: sessions untouched (unread, un-searched, un-cited) for >18mo are "cold." Cold source blobs get level-19 recompression + can migrate to a `cold/` shard subtree that btrbk/backup treats differently. Track last-access via a lightweight touch on read paths. — NEW

- **Quantify lineage dedup savings as a reported number, and extend it to cross-session blob dedup** — the docs claim lineage stores only divergent tails (16K physical vs 8.8K logical, ~32% dup messages) but that saving is asserted, never measured on the live archive. Produce a "dedup dividend" report; then note blob store *already* dedups identical attachments by hash — measure how much (identical images pasted across sessions, repeated tool outputs) to decide if content-defined chunking is worth it. — NEW (grounds swarm-1 "construct-validity-as-substrate")

- **Content-defined chunking (FastCDC) for large near-duplicate blobs before deciding it's not worth it** — full-file SHA dedup misses the common case: a 2MB file re-uploaded with a one-line diff stores twice. For the attachment/large-blob class (not tiny JSON), rolling-hash chunking + a chunk-ref table could cut the referenced-attachment surface materially. Measure first (prior idea), build only if the dividend clears the index-complexity cost. — NEW

- **A "GC-safe-to-drop" classifier that separates reclaimable from irreplaceable at the row level** — GC today is purely reference-counting (delete if no DB ref + no lease). Affordability needs a policy layer above it: which *referenced* bytes are cheap to re-derive (raw provider JSONL still on disk at `~/.claude/projects` → droppable, re-acquirable) vs truly irreplaceable (browser-captured bytes with no live source). Tag blobs with a `reacquirability` class so a future space-pressure sweep drops the re-derivable set first. — NEW (ties polylogue-83u.4 census)

- **Derived-tier rebuild cost budget, measured and surfaced** — index.db + embeddings.db are "free to delete, rebuild from source" *in theory*; in practice a full reindex + re-embed of 4.3M messages is hours of CPU + Voyage API $. Measure and record the wall-clock + dollar cost of a full derived rebuild so "just reset --index" is an informed decision, not a foot-gun. This is the hidden liability in the durability model: rebuildable ≠ free. — NEW

- **Embeddings are the second storage bomb — make them tiered/prunable** — Voyage 1024-dim float32 over 4.3M+ blocks is ~17GB uncompressed of vectors alone and grows linearly forever. Options: int8/binary quantization of `vec0` (4× shrink, small recall loss), embed only human+assistant blocks (skip tool_result/runtime_protocol material — huge fraction of block count), and drop embeddings for cold sessions (re-embed on demand). Embeddings.db is rebuildable, so aggressive pruning is safe. — NEW

- **Retention policy for `ops.db` and convergence_debt / cursor-lag / otlp tables** — ops.db is "disposable" but these append-only telemetry tables grow unboundedly and are never truncated. A rolling window (keep 90d of samples, aggregate older into daily rollups, hard-drop otlp) keeps the disposable tier actually disposable instead of a silent multi-GB accretion. — NEW

- **Raw-snapshot retention should be provider-aware and time-decayed, not fixed-N** — `raw_retention.py` keeps a fixed count of superseded live snapshots per source. For hot-appending Codex/Claude sessions that get re-acquired hundreds of times, fixed-N still hoards near-identical snapshots. Switch to keep-latest + exponential-decay sampling (last 3, then 1/day, then 1/week) so you retain audit trail without linear growth on chatty sources. — NEW

- **Backup surface is dominated by the blob store — make backup incremental over the content-addressed tree** — because blobs are immutable and content-addressed, a backup only ever needs to ship *new* shard files; the address is the dedup key. Verify the backup manifest exploits this (ship-new-only) rather than re-hashing 36GB each pass, and after zstd lands the backup surface drops with the store. This is the difference between a 10yr backup that's O(new bytes) vs O(total). — NEW (ties polylogue-83u.4 backup verifier)

- **Set and enforce a per-origin bytes-per-session SLO with an ingest circuit-breaker** — the operator's CLAUDE.md has a disk-write SLO ethic (60GB/day root, 400GB/day NVMe). Give the archive the same: if a provider's ingest starts averaging >X MB/session (a parser regression, a base64-inline-image explosion), the daemon flags it in `status` rather than silently 10×-ing the store. Affordability is a *guardrail*, not a periodic cleanup. — NEW

- **"Drop-and-account" tombstones so GC'd content stays citable** — the fear that blocks corpus-compaction / cold-drop is "if I delete the bytes, past citations dangle." Store a tombstone row (hash, origin, size, dropped_at, reacquirable-from) when reclaiming, so a citation-anchor still resolves to "this existed, N bytes, re-acquirable from PATH / permanently gone" instead of a hard 404. Makes aggressive retention politically safe. — NEW (grounds swarm-1 "citation-anchors" + "recursive-safety gate")

- **Compression + GC + verify in a single shard walk (already noted in 83u.5 — commit to it)** — the recompression pass, the GC sweep, and the blob-integrity verify all walk the same two-level shard tree. One `ops maintenance blob-compact` job that does verify-hash → recompress → GC-if-unreferenced amortizes I/O and turns three periodic full-tree scans into one. At 500GB this I/O consolidation is the difference between a maintenance job that fits in a quiet window and one that doesn't. — polylogue-83u.5 (synergy already designed; elevate to acceptance criterion)

- **A yearly "corpus compaction" epoch: freeze + repack old years into read-only, maximally-compressed archive segments** — beyond incremental GC, adopt an epoch model: once a calendar year is fully cold, repack its blobs into a single level-19 zstd-dictionary segment + a frozen index shard, mark it read-only, exclude from live convergence. Live archive stays small and fast; history stays queryable but off the hot path. This is the concrete shape of "affordable at 10 years." — NEW (grounds swarm-1 "corpus-compaction" convergent theme)

---

### GPT-pro prompt stubs

**[DR] Compression & retention prior art for a personal append-only content-addressed archive.** "Survey the state of the art (2023-2026) for keeping a multi-year, single-writer, content-addressed local archive affordable: zstd trained dictionaries vs generic frames for small repetitive JSON; FastCDC / content-defined chunking economics vs whole-file dedup; how restic/borg/bup/Kopia model incremental backup over immutable chunk stores; embedding-vector quantization (int8/binary) recall tradeoffs at scale; and time-decay retention sampling schemes. Give concrete compression-ratio and dedup-dividend expectations for provider-chat JSON and give a decision rule for when CDC beats file-hash dedup."

**[DR] Cost of derived-data rebuilds vs storing them: the 'rebuildable tier' fallacy.** "For a system with durable source data and 'rebuildable' derived tiers (full-text index + a 1024-dim embedding index over ~4M records), analyze when 'delete and rebuild from source' is actually cheaper than retaining, factoring re-embedding API cost, CPU-hours, and the operational risk of a multi-hour rebuild. Survey how large systems (search engines, feature stores, vector DBs) budget and amortize reindex cost, and propose a policy for partial/tiered rebuild-on-demand."

**[A] Design the tiering + retention policy engine.** "Design a policy layer over a content-addressed blob store that classifies every blob on two axes — durability (irreplaceable / re-acquirable-from-live-source / re-derivable) and access-temperature (hot / warm / cold by last-read) — and drives a maintenance job that recompresses cold blobs harder, drops re-derivable cold blobs under space pressure leaving citable tombstones, and repacks fully-cold calendar years into frozen read-only segments. Specify the tombstone schema, the space-pressure trigger, and the safety invariants that keep it lease/GC-safe and never touch irreplaceable-hot bytes."

---

## [a307501b00d42c891] LANE 12 — ATTENTION / TRIAGE / THE HUMAN FRONTIER

Grounded in `resume.py`, `find_abandoned_sessions`, `find_stuck_sessions`, `find_resume_candidates`, terminal-state severity ladder, and the `assertions`/marks tier. Here is the lane.

## LANE 12 — ATTENTION / TRIAGE / THE HUMAN FRONTIER

- **A standing `frontier` view (context-free inbox) that ranks ALL 16.7k without a `repo_path`** — `find_resume_candidates` is fatally coupled to the operator's *current* cwd/recent-files; you can't ask "what deserves my eyes today?" from a cold terminal. Invert it: default to a repo-agnostic ranking (recency-decayed terminal-severity × durable-value) so the frontier exists as a homepage, not a context-lookup. — NEW

- **A single `worth_reviewing_score` per LOGICAL session, materialized in index.db** — compose the pieces that already exist but never combine: terminal-severity weight (from `_terminal_weight`), unresolved-blocker count (from `ResumeInferences.blockers`), open-question detection, staleness half-life, and a durable-value prior (workflow_shape ≠ chat). One number, cited by its component breakdown like `ResumeCandidate.score_breakdown` already does — "analytics one measure away." — NEW

- **A `reviewed`/`dismissed` assertion kind so the frontier is a true inbox that empties** — right now nothing removes a session from abandoned/stuck lists; they re-surface forever (attention leak). Add `AssertionKind.triaged` with a verdict (`resumed` | `wont_resume` | `archived` | `snoozed:<until>`); frontier queries `WHERE NOT EXISTS triaged`. This is the load-bearing state that turns detectors into an inbox. — NEW

- **Snooze-with-wake, not just dismiss** — "not today, but nag me in 2 weeks" is the dominant human triage move. A `triaged:snoozed` assertion carries a `wake_at`; the frontier re-promotes on expiry. Prevents the binary "resolve or ignore forever" that makes people stop looking. — NEW

- **Durable-vs-disposable classifier as a first-class axis, not a filter side-effect** — `find_resume_candidates` silently drops `clean_finish`, but disposability is richer: a 4-message `chat`-shape session that reached a clean answer is disposable; a `subagent_dispatch` with 3 touched repos and an unresolved error is durable-high. Materialize a `disposability` label (disposable/ephemeral/durable/keystone) from workflow_shape + tool-category breadth + file_paths_touched cardinality, and let the frontier default to hiding disposable. — NEW

- **"You started this and never finished" = terminal-state + zero-follow-up detector** — cross `find_abandoned_sessions` (question_left/tool_left/agent_hanging) with `session_links`: an abandoned tail with NO child/resume edge and no later session touching the same `file_paths_touched` is genuinely dropped work. An abandoned session that WAS silently continued elsewhere is *not* frontier-worthy. This lineage join is the difference between a real loose end and noise. — NEW

- **Batch related loose ends by thread/repo into "clusters," not rows** — the operator triages *topics*, not sessions. Group frontier items by `ThreadInsight.thread_id` and dominant_repo (both already in `resume.py`), so "the polylogue lineage work has 4 abandoned tails across 3 days" is ONE frontier card with a rolled-up next_step, not 4 competing rows. Corpus-compaction applied to attention. — NEW

- **Surface the actual open question, not just "question_left"** — `find_abandoned_sessions` reports terminal_state but the human needs the *text*: the last unanswered user/assistant question verbatim, with a citation-anchor (message_id) so a click jumps to it. Extract from `_last_message` when `role==user` (already computed in `_next_steps`) and from question-shaped terminal evidence. Attention needs the sentence, not the label. — NEW

- **A `triage` CLI verb / `read --view frontier`** — the human surface. `polylogue triage` prints the top-N frontier cards (score, one-line why, open-question text, cluster, suggested action), each with a `mark reviewed`/`snooze 2w`/`continue` affordance inline. Reuses the query-first grammar (`triage repo:polylogue since:14d unresolved:true`) so the frontier is a *saveable query object*, not a bespoke report — builds directly on wave-1 "queries/findings-as-objects." — NEW

- **Blocker-centric frontier lane: "these N sessions are all blocked on the same thing"** — `ResumeInferences.blockers` is per-session prose today; normalize/cluster blocker strings (embedding-similarity via the existing embeddings.db) so "3 sessions blocked on the same missing migration" collapses into one actionable item. Unblocking one often unblocks many — highest attention-ROI signal in the archive. — NEW

- **Decay curve tuned to human memory, not recency** — `find_resume_candidates` uses a fixed 72h half-life for *resumption* relevance, but *review-worthiness* decays differently: a high-value abandoned session gets MORE urgent for ~2 weeks (still recoverable in your head) then drops sharply (context evaporated, becomes archaeology). Model an inverted-U attention-urgency curve, distinct from resume-recency. Construct-validity: name the thing you're actually measuring. — NEW

- **Stuck-session frontier promotion requires a live-vs-dead check** — `find_stuck_sessions` bounds tool-call latency, but a "stuck" session that's actually a still-appending hot Codex file (the daemon's quiet-deferral case) is NOT frontier-worthy — it's just in-flight. Gate stuck-promotion on "no new messages in >quiet-window AND terminal," reusing the daemon's hot-file signal, so the frontier never nags about work happening right now. — NEW

- **Recursive-safety gate on the frontier: never surface a session about triaging the archive itself** — dogfooding means R&D/meta sessions (like this swarm) will score high on "unresolved + durable." Tag `material_origin`/repo=polylogue meta-work and let the frontier de-prioritize self-referential archive-maintenance sessions unless explicitly asked — otherwise the inbox fills with its own reflection. — NEW

- **"Frontier digest" as a recall_pack the operator can pin** — persist today's frontier as a dated `AssertionKind.recall_pack` (durable user.db) with citation-anchors, so "what did I decide to skip last Tuesday and why" is answerable, and dismissals become an auditable trail rather than silent loss. Turns triage decisions into durable findings-as-objects, and lets the ambient-notification agent *read* the frontier instead of recomputing it. — NEW

- **A `next_action_type` label on each frontier card (decide / resume / review / discard)** — `_next_steps` already emits prose next-steps; classify them into a small action vocabulary so the operator can filter "just show me DECISIONS I owe" vs "resumable coding work." Different attention modes want different subsets; one undifferentiated list is itself an attention tax. — NEW

- **Frontier honesty: show the confidence and the uncertainty, don't launder it** — terminal_state and blockers carry `confidence`/`support_level`/`uncertainties` fields (already in the models) that get dropped at every summary surface. A low-confidence "abandoned" guess should render *as* a guess ("possibly unfinished — weak signal"), so the human's trust in the frontier is calibrated. A frontier that overclaims gets ignored within a week. — NEW

## GPT-pro prompt stubs

- **[A]** "Design a per-session `worth_reviewing_score` for a personal AI-session archive (16.7k sessions, SQLite). Inputs available per logical session: terminal_state ∈ {clean_finish, question_left, error_left, tool_left, agent_hanging, unknown} with a confidence; workflow_shape ∈ {chat, planning, implementation, debugging, agentic_loop, subagent_dispatch}; last_message_at; count of unresolved blockers (prose); count of distinct files touched; whether a lineage edge shows the work continued elsewhere; days since last touch. Produce (a) a decomposable scoring formula with per-component weights and a stated rationale, (b) an inverted-U 'attention-urgency vs. staleness' curve distinct from resume-recency, and (c) the boundary conditions that should force score→0 (disposable, superseded, in-flight). Justify every weight; flag construct-validity risks where the signal doesn't measure what the name claims."

- **[A]** "I'm building a triage 'inbox' over a read-only archive of past AI coding sessions. Detectors surface abandoned/stuck/unfinished sessions, but the list never empties and re-nags forever. Design the minimal state model that turns detector output into a real inbox: triage verdicts, snooze-with-wake semantics, cluster-level (topic/thread) dismissal vs per-item, an audit trail of what was skipped and why, and interaction with the fact that the underlying sessions are immutable and new detections keep arriving. Specify the state transitions and the exact query that computes 'today's frontier.'"

- **[DR]** "Survey how mature tools solve human triage of a large, ever-growing queue of loosely-structured items where most are ignorable and a few are high-value: email inboxes (Superhuman/HEY 'the imbox'), bug/issue triage rotations, security-alert fatigue and SOC alert de-duplication, RSS/read-later apps, and code-review queues. Extract the concrete mechanisms — scoring, snooze/defer, clustering related items, 'zero-inbox' pressure, staleness decay, dismissal audit trails, distinguishing durable from disposable — and map each to a design for an operator-facing 'frontier' over 16.7k archived AI sessions. Cite sources and call out which mechanisms fail when items are immutable and detection is heuristic."

---

## [a193004fdf9a31ad2] Grounded in `user.py`/`user_write.py` (unified `assertions` table: `scope_ref`/`target_ref

Grounded in `user.py`/`user_write.py` (unified `assertions` table: `scope_ref`/`target_ref`/`evidence_refs_json`/`supersedes_json`/`context_policy_json`/`staleness_json`/`confidence`/`author_kind`; `AssertionKind` already carries `SAVED_QUERY`/`RECALL_PACK`/`WORKSPACE_NOTE`/`DECISION`/`BLOCKER`/`LESSON`/`JUDGMENT`) and the MCP surface (`save_saved_view` stores a flat `query_json` blob; `session_work_events`, `blackboard_post`, `get_resume_brief` exist). The frontier: saved queries are inert strings today — no id-as-ref, no result-set memory, no query→query edges, no self-tracking of the archive's own use.

- Promote saved queries to a first-class `query:<id>` ObjectRef kind — so any assertion (mark/note/correction/blocker) can `target_ref` a query and any `evidence_refs_json` can cite one; today a saved_query is an assertion but nothing can *point at* it, so a query can't accrue annotations, judgments, or supersession. Unlocks "this query is the canonical way to find X" as durable knowledge. — NEW (extends polylogue-37t)
- Materialized query result-set snapshots with content-hash in ops.db — on each execution persist `(query_id, frontier_session_ids, result_hash, ran_at, ran_by_session)`; the result set becomes a diffable stored object so `read query:abc --diff` shows "3 sessions entered / 1 left since last run" and assertions can attach to a *frozen* answer ("these 12 were the answer on 06-29"). — NEW
- Query-reference edges + dependency invalidation (storage layer, not grammar) — persist a `(parent_query, child_query)` edge table when one saved query is defined over another, so deleting/renaming a base query surfaces dependents and marks their cached result-sets stale; the hierarchical query→result-set DAG the zone asks for, kept at the object/reference layer to stay out of the DSL-grammar exclusion. — NEW
- Standing queries as change-detectors that emit CANDIDATE assertions — a saved_query whose `context_policy_json` says `{watch:true}` re-runs on the converger's quiet window; a delta in its result set writes an `AssertionKind` candidate ("new stuck session matched standing query 'abandoned in polylogue repo'") for operator triage. Turns inert views into the archive's own alerting. — NEW (feeds polylogue-rii)
- Self-mine BLOCKER/LESSON/DECISION assertions into discovered beads — assertions authored during dev sessions are latent tracked debt; a promotion pass clusters same-target blockers/lessons and proposes `bd create` payloads with `evidence_refs` → the originating session, converting anonymous in-archive knowledge into ready work. (Creation from assertions, distinct from bead-*audit*.) — NEW (polylogue-37t)
- Recursive-safety gate: agent-authored assertions default to CANDIDATE + `inject:false` — `author_kind` already distinguishes user vs agent; enforce that anything an agent writes while *reading* the archive cannot re-enter context until operator-promoted. This is the structural fix for the recovery-report-fabrication failure mode (archive feeding its own hallucinations back). Non-obvious and load-bearing for the self-referential loop. — NEW (polylogue-7aw)
- Ingest polylogue's own dev sessions into a distinguished `self` workspace — auto-tag sessions whose cwd-prefix == POLYLOGUE_ROOT so the archive can answer "what did we decide about the context scheduler?" over its own construction history, and so self-queries have a clean scope. The recursion made addressable. — NEW
- Read-access log as a queryable unit source — persist every MCP/CLI read (`get_session`, `search`, recall-pack open) as a lightweight ops.db event; expose "most re-read sessions", "saved queries that return nothing", "recall packs never opened" (dead-memory detection). The archive tracking its own use, and the raw signal the context scheduler needs. — NEW (polylogue-37t)
- Context scheduler: budget-aware inject selection ranked by staleness × attention × topic-proximity — `context_policy_json` today is a boolean; add a selector that, given a token budget and the current session's topic embedding, picks which inject-eligible assertions to surface, decaying by `staleness_json` and boosting by embedding proximity + recent read-log attention. Agent memory reboot with a real ranking function instead of dump-everything. — NEW (polylogue-37t)
- Programmable `context_policy_json` windows — extend `{inject:bool}` to `{inject, when:[session_start|on_topic|on_error], ttl_days, max_injections, cooldown}` so an assertion becomes a scheduled memory primitive (a caveat that fires only on error, a lesson that surfaces once then sleeps). The scheduler above honors it. — NEW
- Annotation recipes: parameterized assertion templates — a `metadata` assertion storing a recipe that, given a session id, runs sub-queries + templated bodies to emit a coherent set (e.g. a "postmortem recipe" auto-attaches decision/blocker/lesson skeletons pre-filled from `session_phases`/`get_pathologies`). Annotation-recipe substrate; makes structured curation one call instead of ten. — NEW
- Belief-history read over `supersedes_json` — a `belief_timeline(target_ref)` insight reconstructing how a decision/correction evolved ("thought X → corrected to Y after session Z → re-affirmed X"), so an agent reboot sees the *reversal history*, not just the current row, and stops re-litigating settled reversals. Uses a field already present but never surfaced. — NEW
- Live recall packs bound to a saved query — today a recall_pack is a frozen session list; bind it to a `query:<id>` so `get_resume_brief` recomputes *membership* live while preserving the human narrative + per-session annotations. Marries queries-as-objects to the continuity surface. — NEW (extends existing recall-pack surface)
- Semantic dedup of saved queries via embeddings — embed saved-query bodies into embeddings.db so an agent asks "is there already a view for this?" before minting a 40th near-duplicate, and clustering proposes canonical named views. Prevents the queries-as-objects store from silently rotting into noise. — NEW
- Reverse "annotation density" insight (session → assertions-about-it) — a materialized read exposing "most-annotated sessions", "sessions with unresolved blockers", "sessions with zero human touch" — makes the archive's own knowledge-accretion visible and steerable, and gives the scheduler a signal for which sessions carry curated weight. — NEW (polylogue-rii)
- Blackboard posts carry query refs, not just text — `blackboard_post` becomes an inter-agent work channel where a coordinator posts `query:<id>` and workers run-and-report, so a saved query is a unit of dispatchable coordination work. Small change, turns the blackboard from a notepad into a task bus. — NEW (polylogue-s7ae)
- Synthetic "meta-session" of the archive's own runtime — project ingest/convergence/read events as blocks in one synthetic session so the *entire* read-model stack (transcript view, `session_phases`, tool-timing, pathology mining) applies to the archive operating on itself; dogfoods every insight against polylogue's own behavior with zero new insight code. Bold, maximally self-referential. — NEW

GPT-pro prompt stubs:

- [A] "Design the storage + reference model to turn Polylogue saved queries (currently a flat `query_json` blob inside a `saved_query` assertion) into first-class objects: a `query:<id>` ObjectRef kind, content-hashed result-set snapshots (which tier — ops.db disposable vs index.db rebuildable?), a query→query dependency edge table with invalidation, and standing-query change-detection emitting CANDIDATE assertions. Given the unified `assertions` table (scope_ref/target_ref/evidence_refs_json/supersedes_json/context_policy_json) and the five durability-keyed tiers, specify DDL placement, invariants, and the read/write API without touching the query DSL grammar. Enumerate failure modes for the recursive loop (archive querying itself)."
- [A] "Specify a context scheduler for Polylogue: given a token budget, a current session topic embedding, a read-access log, and `assertions` rows with `context_policy_json`/`staleness_json`/`confidence`/`author_kind`, define the ranking function and the programmable-window policy schema (`when`/`ttl`/`max_injections`/`cooldown`). Include the recursive-safety gate that keeps agent-authored assertions out of injected context until operator promotion, and prove it prevents self-hallucination feedback."
- [DR] "Survey how existing systems treat persisted queries as first-class, referenceable, versioned objects with materialized/diffable result sets and change-detection — datalog/materialized-view engines, notebook/dbt lineage, Datasette/Steampipe saved queries, Kùzo/graph views, event-sourced read models, and 'living document' knowledge bases. Extract concrete patterns for query→query dependency invalidation, result-set snapshotting, and standing-query alerting applicable to a single-writer SQLite personal archive."

---

## [a7ffbb35b297eddf2] PROACTIVE / AMBIENT SURFACING — 15 ideas

Grounding done. Key finding: Polylogue already has a full notification substrate (5 backends: log/webhook/journald/email/apprise via `FanOutNotificationBackend`), a `daemon_events` ledger in ops.db, SSE push (`events_http.py`, `text/event-stream`), tiered `HealthAlert` severities, and `find_stuck/abandoned_sessions` + `find_resume_candidates`. The gap my lane exploits: that pipe today carries only **operational** health alerts (disk, cursor-lag, convergence-debt), never **semantic/content** events. Everything below routes content signals through the existing fan-out instead of building a new channel.

---

**PROACTIVE / AMBIENT SURFACING — 15 ideas**

- **Semantic alert tier on the existing `HealthAlert` bus** — add a `Notice` severity below WARNING and a `content` alert-family so `find_stuck_sessions`/abandoned/pathology results ride the *same* fan-out (journald/apprise/webhook) as disk-space alerts. The pipe exists; it just never carries meaning. — reuses proven delivery + rate-limiting, zero new channel — NEW

- **Standing queries as `daemon_events` producers, not a new engine** — persist saved-views (assertion `saved_query`) with a `notify_on: appeared|disappeared|count_crossed` policy; converger re-evaluates them post-ingest and writes a `standing_query.hit` row to the ledger. Findings-as-objects (wave-1) become *pushed* objects. — turns the query DSL into a subscription surface for free — NEW

- **Digest is a query over `daemon_events`, not a re-scan** — a `polylogue brief --since 24h` reads the event ledger (abandoned, stuck, pathology, cost-spike, standing-query hits) and renders one capped digest, matching the operator's `oracle` habit but sourced from real events instead of `claude -p` re-summarizing raw logs. Cheap, deterministic, citation-anchored. — folds Polylogue into the existing daily-digest ritual — NEW

- **Three-cadence delivery, one policy table** — on-event (CRITICAL only: cost anomaly, convergence-debt stall), daily-rollup (abandoned threads, stuck sessions), weekly-retro (pathology trends, repeated-mistake clusters). A single `notification_policy` assertion kind gates which family goes to which cadence/backend. — one knob prevents the classic "everything is on-event" fatigue collapse — NEW

- **Terminal MOTD via the SessionStart recall hook** — the operator already has `sessionstart-polylogue-recall.sh`. Extend it to print a 3-line ambient strip: N unfinished threads in this cwd, 1 abandoned >7d, last cost-anomaly. Push at the exact moment a coding session opens. — hijacks an existing hook; zero new daemon surface — bead-adjacent to recall hook — NEW

- **Waybar/desktop-notify backend as a 6th adapter** — an `apprise`-config for `dbus://` desktop notifications plus a `--json` MOTD endpoint Waybar polls (`daemon /api/ambient`). Keep it a thin backend so it inherits the fan-out's per-backend failure isolation. — ambient peripheral awareness (raw-log thread) without terminal focus — NEW

- **"You're repeating a past mistake" real-time nudge** — on ingest of a live Codex/Claude session, embed the current tail and `find_similar_sessions` against sessions tagged `pathology`/`lesson`/`blocker` (assertion kinds already exist). If cosine > threshold AND the past session ended abandoned/errored, emit a `Notice` with a citation-anchor to the prior failure. — the single highest-value ambient signal; grounded in real assertion vocabulary + embeddings.db — NEW

- **Alert fatigue control = per-family token bucket + suppression assertions** — reuse the email backend's existing token-bucket, generalize it per alert-family, and honor `AssertionKind.SUPPRESSION` so "stop telling me about session X" is a first-class durable mute, not a config edit. Snooze = suppression with a TTL. — construct-validity: a suppressed signal is *recorded as suppressed*, auditable — NEW

- **Recursive-safety gate on the notification path** — the daemon ingests its *own* MCP-injected preambles and agent sessions; a nudge that fires on a session *about* a past nudge is a feedback loop. Gate: never emit content-alerts for sessions whose `material_origin` is `generated_context_pack`/`runtime_context`, and dedupe by citation-anchor so the same finding never re-notifies. — directly builds on wave-1 recursive-safety theme — NEW

- **MCP-injected ambient preamble as a delivery channel** — `compose_context_preamble` already exists; add an `ambient` section that injects the current unfinished-thread list + top repeated-mistake warning into an agent's opening context. The archive becomes a collaborator that *briefs the next agent* before it starts. — continuity surface is MCP, not CLI (memory doctrine) — NEW

- **Unfinished-thread surfacing keyed to cwd + repo** — `find_abandoned_sessions` filtered by the current project's `cwd-prefix` so the nudge is contextual ("you have 2 open threads *in this repo*"), not a global inbox. Rank by recency×investment (message count × tool-calls), surface top 3. — relevance gating is the entire anti-noise lever — partially bead-9e5 adjacent — NEW

- **Cost-anomaly as an on-event push, not a monthly surprise** — `cost_outlook`/`session_costs` already compute per-session spend; add a converger stage that fires a `Notice` when a live session's rolling cost crosses a z-score threshold vs. that model's baseline. Ties to the known credit-rate/token-double-count history — anomalies are the early-warning for those bugs recurring. — analytics one-measure-away (wave-1) → pushed — NEW

- **Digest honesty: every ambient claim carries a citation-anchor + confidence** — reuse the recovery-digest lesson (fabricated "PR #123 merged" from regex text-mining). Ambient notifications must cite `session_id:message_id` and mark text-mined vs. structural-fact provenance. No unanchored claim leaves the daemon. — construct-validity-as-substrate; hard-won from the 2026-06-29 fabrication incident — NEW

- **"Now-quiet" deferral for nudges, mirroring convergence hot-file deferral** — the converger already batches still-appending sessions until a quiet window. Apply the identical primitive to notifications: never nudge mid-session; wait for the quiet-window close, then emit the retro. Prevents interrupting active flow. — reuses `convergence_stages.py` quiet-deferral logic verbatim — NEW

- **Ambient-event replay + "what did you not tell me" audit** — because every notification is a `daemon_events` row, add `polylogue brief --replay --since 7d` to see the full stream *including suppressed/deferred* events. Answers "is the archive silently sitting on something?" and lets the operator tune thresholds against real history. — makes the ambient layer itself auditable, not a black box — NEW

---

**GPT-pro prompt stubs**

- **[A]** "Design a semantic notification policy layer for a single-user local archive daemon that already has a fan-out backend bus (log/journald/email/apprise/webhook) carrying only operational health alerts. I need: (1) an event taxonomy separating operational vs. content-semantic signals; (2) a per-family cadence model (on-event/daily/weekly) with a token-bucket + durable suppression/snooze store; (3) an anti-fatigue scoring function that ranks 'worth interrupting' vs. 'batch into digest'. Constraints: every alert must carry a provenance/confidence tag; no alert may fire on the daemon's own generated context. Give me the policy schema, the scoring formula, and the failure modes."

- **[DR]** "Survey how ambient/proactive information systems control notification fatigue while staying useful: research on interruptibility/attention (Horvitz, notification-management), digest-vs-realtime tradeoffs, 'peripheral awareness' UI patterns, and real-time 'you're repeating a past mistake' nudging (spaced-repetition-of-mistakes, IDE warning fatigue). Deliver concrete thresholding heuristics, empirical fatigue-onset findings, and design patterns for a single-user developer tool that pushes signals from a personal-history archive."

- **[DR]** "Compare architectures for turning saved queries into standing subscriptions/change-detectors over an append-only event ledger (SQLite `daemon_events`-style): materialized-view diffing, CDC/trigger approaches, and periodic re-evaluation with appeared/disappeared/count-crossed semantics. Focus on: dedup by stable finding-identity, avoiding re-notification storms after backfills/re-ingests, and cheap idempotent evaluation on each ingest tick. Cite real systems (incremental view maintenance, Materialize, Debezium, RSS-diff/changedetection tools) and give a recommended minimal design."

---

## [a3d370a21657d580f] Grounded in rii (live write-leg/read-back), 37t (claims→judgment→preamble→reboot + 37t.12 

Grounded in rii (live write-leg/read-back), 37t (claims→judgment→preamble→reboot + 37t.12 judgment queue), s7ae/kph (coordination + provenance-carrying PRs), jlme/3v1.1/83u.3/90y (browser capture), and MEMORY (GPT-pro auto-capture, `g-p-` project id, recovery-digest honesty, assertions > CLAUDE.md). Lane #5 — the self-capturing recursive loop as designed system:

- **Synthetic runtime meta-session** — project daemon convergence events, ingest cursors, embed runs, `convergence_debt` (ops.db) into a first-class session/message/block tree so the archive's own runtime is queryable in the same algebra as chats (`find "spool stall" origin:polylogue-runtime`). The loop can only be governed if the machine's own heartbeat is a retrievable object, not just logs. — NEW (feeds rii self-analytics)
- **R&D cohort as a first-class object** — auto-mint a cohort ref for each agent wave (session-tree ∪ repo ∪ time-window), so "wave-2 self-capture, 2026-07-05" is one addressable set the operator judges/diffs/cites as a unit. Self-capture means the archive holds its own genesis; make each batch retrievable. — NEW (extends s7ae envelope)
- **Design-chat → bead → PR provenance chain** — the GPT-pro chat that proposed a feature is auto-captured under its `g-p-` project id; wire it as the authoring-provenance so `resolve_ref` on a bead walks back to the originating design conversation, and PR postmortems (kph) attach the design chat, not just the coding session. Closes capture→distill→execute→capture into a traversable arc. — extends kph + rii
- **Closed-loop laundering gate (THE invariant)** — any agent-authored assertion whose evidence citations resolve *only* to other agent-authored sessions in this archive gets a `closed-loop` flag → forced candidate + inject:false until a human judgment or an externally-grounded citation (source.db raw bytes, git, external doc) breaks the cycle. This is what stops the archive re-injecting its own hallucinations as retrieved "evidence." — extends 37t.12 + recursive-safety
- **Hallucination-provenance cohort** — when operator judgment rejects an agent assertion, persist the rejection *plus the fabricated citation-anchor*; accumulate a pathology cohort of the archive's own false claims so retrieval can down-weight sessions that historically laundered fabrication (recovery-digest "PR #123 merged" class). Self-capture uniquely lets the system learn its own failure signature. — extends pathology mining + construct-validity
- **Self-mined candidate beads (never auto-filed)** — run self-analytics over the R&D cohort to surface follow-ups agents wrote in prose ("we should…", "follow-up:") but never filed; emit as candidate beads carrying a mandatory citation-anchor to the source block, staying in candidate state until judged. Honest text-mining = candidates, not facts. — NEW child under rii/37t.10
- **Loop-latency as a native measure** — the archive already holds all four timestamps: design-chat ingest → bead filed → PR merged → postmortem re-captured. Surface end-to-end R&D cycle time as a cost/latency projection ("analytics one measure away"); no new capture needed, only a join. — NEW (extends session_latency_profile)
- **Cohort-scoped preamble carry** — when a new wave spawns, `compose_context_preamble` injects the *prior wave's judged findings-as-objects* (not raw transcripts), filtered through the closed-loop gate. Each wave stands on the accepted distillate of the last, so recursion compounds knowledge instead of re-litigating it. — extends 37t.11/37t.4
- **Cohort drift/repetition diff** — embedding-diff wave-N against wave-(N-1) and against the rejected-assertion cohort to flag agents re-proposing already-killed ideas; a "novelty score" per cohort keeps the loop from spinning in place. Self-capture makes the archive its own regression baseline. — NEW (uses find_similar_sessions)
- **Judgment queue batched BY cohort** — 37t.12 should present candidate assertions grouped per R&D wave so the operator judges a whole batch's distillate in one pass rather than a trickle; makes the human-in-loop the governor of recursion at human scale, not a bottleneck per-claim. — extends 37t.12
- **Reboot-with-refs reads its own tail** — an R&D batch that exhausts context reboots into a compact evidence pack of *its own just-captured turns* pulled live from the archive (rii read-leg + 37t.3): the agent literally reads back what it just wrote, the tightest possible self-capture loop, and the proof that write-leg + read-leg cohere. — grounds rii + 37t.3
- **Runtime-debt as self-healing feedstock** — expose `convergence_debt`/cursor-lag so a self-analytics agent can locate where the archive is failing to ingest its *own* R&D (GPT-pro chat stuck in spool, cohort session unmaterialized) and file a candidate repair bead. The system audits its own capture completeness. — extends synthetic meta-session
- **In-page overlay shows the live recursion** — 90y overlay on chatgpt.com surfaces "this chat is captured AND feeding beads X,Y,Z" while you design; the operator sees the loop closing in real time and can veto/annotate at the source. Makes the recursion legible at the point of authorship. — extends 90y
- **Design-artifact byte capture** — GPT-pro design chats paste architecture screenshots/diagrams; 83u.3's byte-acquisition path means the archive holds the *visual* design provenance (real blob, nonzero bytes), not chip text — the diagram that spawned the schema is as retrievable as the prose. — grounds 83u.3
- **Self-capture dedup integrity** — concurrent capture instances (3v1.1) + lineage replay means a design chat can be captured multiple times and replayed as fork prefix; the content-hash dedup must treat the archive's own R&D captures as first-class or cohorts double-count velocity/cost. Loop integrity depends on honest dedup of self. — grounds 3v1.1
- **NEW capability unlocked — counterfactual replay** — because the archive holds its own dev sessions AND runtime meta-sessions, you can replay "what context did the agent actually have when it made decision D?" by reconstructing the preamble + retrieved set at that timestamp, and test whether a better-judged distillate would have changed the outcome. Only self-capture makes the system's own decisions auditable against its own evidence state. — NEW (extends counterfactual/replay tooling)

GPT-pro prompt stubs:

- **[A]** "Design the recursive-safety invariant for an archive that ingests its own AI-authored R&D. Given assertions with `author_ref` and citation-anchors, specify the exact predicate that flags a 'closed-loop' claim (evidence resolving only to other agent-authored sessions), the quarantine state machine (candidate + inject:false → break conditions), and how operator judgment or external-grounded citations release it. Show the SQL-expressible check and the failure mode if the predicate is too loose vs too tight."
- **[A]** "Propose the schema for projecting an archive's own runtime (daemon convergence events, ingest cursors, convergence_debt, embed runs) into a synthetic session/message/block meta-session queryable in the same algebra as chats. What are the identity/hashing rules, what's the retention tier, and what new self-analytics queries does this unlock that logs cannot?"
- **[DR]** "Survey prior art on systems that ingest and act on their own operational history without amplifying their own errors (self-training data loops, model-collapse mitigation, autonomous-agent memory poisoning, provenance/taint tracking, retrieval down-weighting of self-generated content). Extract concrete invariants and failure modes applicable to an AI-session archive that captures its own R&D and re-injects distilled findings into future agent context."

---

## [af4a27b49e2351df4] THE NON-ENGLISH / ACCESSIBILITY OPERATOR — 15 ideas

THE NON-ENGLISH / ACCESSIBILITY OPERATOR — 15 ideas

- Natural-language axis is missing entirely — `languages_detected` (insights/archive_models.py) tracks *programming* languages only (python, rust); there is no per-message/session detected human language, so Polish prose is invisible as a filterable/renderable dimension. Add a derived `prose_lang` signal (fastText/lingua on `human_authored` blocks) → enables `lang:pl`, per-language stats, and reply-language memory — NEW

- FTS unicode61 does not fold Polish stroke letters — `tokenize='unicode61'` (index.py:290) uses default `remove_diacritics 1`, which strips *combining* marks (ą→a, ż→z, ó→o) but NOT precomposed stroke letters ł/Ł (U+0142, distinct codepoints). So "kanał" and "kanal" tokenize differently and never cross-match; a whole letter-class of Polish words has split recall — NEW

- No Polish lemmatization/stemming → inflectional recall collapse — porter isn't compiled (English-only anyway) and Polish is heavily inflected (sesja/sesji/sesję/sesjach are 4 distinct tokens). bm25 ranks fine mechanically but term-frequency is fragmented across cases; searching one form misses the other six. Needs Morfologik query-expansion or a fts5 `trigram` fallback lane — NEW (fold into search-quality bead)

- Query-side diacritic/ł normalization must match index-side — the Lark DSL (archive/query/expression.py) must fold the query token the same way unicode61 folds the index (NFC + diacritic strip + ł→l), or a user typing "różne"/"kanał" gets zero-vs-partial hits depending on which side folds. Today there's no evidence the DSL applies unicode61-equivalent folding — NEW

- No Polish stopword handling — bm25 has no Polish stopword list, so ultra-frequent function words ("i", "w", "na", "że", "się") get full idf weight and dominate/pollute ranking on Polish queries. English-tuned FTS assumptions silently degrade Polish precision — NEW

- Cross-language semantic bridge is unverified for Polish→English — default embedding model is `voyage-4` (config.py:246); the multilingual/vector lane is the *only* realistic path for a Polish query to hit English code/tool-output content, but nothing confirms voyage-4 embeds Polish well or that short Polish queries land near English neighbors. Needs a Polish→English recall probe before relying on it as the fallback — NEW

- DSL tokenizer robustness on non-ASCII unquoted words — the strict command floor (#1842) treats a bare unquoted plain word as UsageError; a bare Polish word (`polylogue sesja`) and field-syntax with Polish values (`repo:moduł`) must survive Lark tokenization without mangling non-ASCII. Confirm the grammar's word/token classes are Unicode-aware, not `[a-z]`-scoped — NEW

- Operator reply-language preference has no home — raw-log states the operator sometimes wants models to reply in Polish, but nothing persists this. A per-session/global `reply_language` in `user.db` assertions (metadata kind) + surfaced in `compose_context_preamble`/recall packs would let continuity carry "operator prefers Polish replies" into injected context — NEW

- Color-blind-unsafe status encoding — status/cost rendering leans on red=error / green=ok (the single worst CVD pair). `NO_COLOR` and `force_plain` exist (config.py) but drop *all* signal; add shape/text glyph prefixes (✗/✓, ERR/OK) so meaning survives both color-blindness and monochrome, not just a color toggle — NEW

- Plain mode ≠ screen-reader mode — `POLYLOGUE_FORCE_PLAIN=1` disables color but Rich still emits box-drawing tables/rules that screen readers verbalize as noise ("horizontal line horizontal line…"). A true a11y/`--machine` mode should emit tab/newline-delimited linear text (or lean on the JSON envelope) with no box glyphs — NEW

- Reader-aloud (TTS) surface can exploit material_origin — raw-log wants voice; `material_origin` already classifies runtime_protocol/tool_result/runtime_context rows, so a "read this transcript aloud" mode can skip code/tool-noise and voice only human_authored + assistant_authored prose with speaker labels. Accessibility payoff and it's a near-free composition over existing enums — NEW

- Voice-in (STT) query path as accessibility, not gimmick — a dictated Polish query hitting the query-first CLI/MCP is the keyboard-free entry point; ties directly to the ł/diacritic/lemmatization gaps above (STT output will be fully-accented Polish, exercising exactly the recall holes). Scope as an MCP/daemon endpoint that normalizes then runs the DSL — NEW

- Phone/mobile reader is unproven — the HTTP daemon (`daemon/web_shell_reader.py`) and `render pages` site are the only non-terminal readers; confirm viewport meta, responsive width, tap-target sizing, and readable font on a phone. A transcript that only reads well at 120 cols is inaccessible to the operator-on-phone — NEW

- Terminal-width robustness for long Polish words — Polish compounds/inflections are long and Rich wrapping + no hyphenation can overflow narrow/phone-SSH terminals; ensure `COLUMNS`/tty-width is honored end-to-end and wide-char/emoji width in mixed prose doesn't miscompute column math (the encoding-boundary matrix already probes CJK/zero-width — extend to Polish) — NEW

- Polish collation for titles/sort — SQLite default BINARY collation mis-orders Polish (ł, ż, ó sort by codepoint, not alphabet), so any alphabetical session-title/tag listing is subtly wrong for the operator's own language. Low-severity but pure assume-English; an ICU/custom collation on display-sort paths fixes it — NEW

---

GPT-pro prompt stubs

[DR] "SQLite FTS5 for Polish full-text search: given `tokenize='unicode61'` with default `remove_diacritics 1` over mixed Polish-prose/English-code content, characterize concrete recall failures — precomposed stroke letters (ł/Ł) not folded, no lemmatization for a heavily-inflected language, no Polish stopword list, bm25 term-frequency fragmentation across inflected forms. Compare remediation lanes: unicode61 `remove_diacritics 2` + custom ł mapping, the fts5 `trigram` tokenizer, an ICU tokenizer build, Morfologik-based query expansion, and offloading Polish recall to a multilingual embedding (voyage-4) vector lane. Give a decision matrix keyed on recall/precision, index size, build-dependency cost, and query-side normalization symmetry."

[A] "Design a natural-language (human prose) detection + tagging layer for an AI-session archive whose operator writes mixed Polish/English (Polish prose, English code/tool-output), keeping it distinct from the existing programming-language detector. Specify: per-block vs per-message granularity, handling of code-fenced regions and quoted tool output, confidence thresholds, and how a derived `prose_lang` signal should surface as a query filter (`lang:pl`), a per-language stats axis, and a persisted operator `reply_language` preference injected into agent context."

[A] "Propose a terminal + phone accessibility contract for a query-first CLI/MCP/HTTP archive built on Rich: a true screen-reader/`--machine` mode (linear, box-glyph-free), color-blind-safe status encoding that degrades to glyphs/text under NO_COLOR, TTS read-aloud that filters by an authoredness enum (skip tool_result/runtime rows), STT query intake, and a responsive mobile transcript view. Prioritize by accessibility impact vs implementation cost and flag which items are pure composition over existing primitives."

---

## [acf18e69afc12eaa4] WHOLE-PRODUCT & GO-TO-MARKET STRATEGY — 16 ideas (grounded in the 3tl legibility epic + ra

WHOLE-PRODUCT & GO-TO-MARKET STRATEGY — 16 ideas (grounded in the 3tl legibility epic + raw-log: Karpathy interest, "prove utility," parents/funding/resume-gap context, "awesome personally useful thingy, likely very impressive," "README still sucks," install-instructions gap, "great demo" framing).

- **Name the true identity: personal local-first substrate that is OSS, never SaaS** — single-writer + all-local architecture makes "hosted product" a category error; the honest thing is an open personal-memory substrate, and pretending otherwise wastes effort on billing/multi-tenancy that will never ship — NEW
- **Treat the public artifact as a *capability proof*, not a product launch** — raw-log is explicit that the real conversion event is a stranger/parent/funder concluding "this person builds serious systems" (resume-gap, parents-deal, satisfaction); GTM should optimize for that judgment, not for installs or MRR — NEW
- **Make the Minimum Viable Public Artifact = the 3tl finding-lane rendered as a static, cold-reader-legible site** — the legibility epic is already building live-cited findings under the 3tl gate; publishing that *is* the demo, and it's the one asset that proves rigor without the viewer trusting you — polylogue-3tl
- **Lead the narrative with "agents that read their own history" (unhobbling), not "chat archiver"** — the raw-log's stated core value is agents recovering context unprompted; continuity/recall via MCP is the differentiated, taste-defensible idea, whereas "export to Markdown" reads as commodity — NEW
- **Target first 10 users as agent-tooling tinkerers with large ~/.claude + ~/.codex corpora** — they already feel "agents are absurdly hobbled regarding history"; a general ChatGPT-export audience doesn't share the pain and can't validate the hard ideas — NEW
- **Distribution wedge = a written data-story essay ("what I found mining 38GB of my own agent sessions"), tool as the CTA** — the findings-as-objects/measure-algebra work produces genuinely novel, live-cited claims; an essay travels in the Karpathy-adjacent "AI memory / context engineering" discourse where a bare Show-HN would not — NEW
- **Ship the multi-agent coordination proof (s7ae) as the flagship live demo** — "two agents on one repo, separate worktrees, visible overlap + scoped handoff" is a jaw-drop artifact for exactly the audience that matters, and the acceptance line already demands a reproducible run script + captured JSON — polylogue-s7ae
- **Close the whole-product wrapper gap: one-command install for nix AND pipx AND a normie-distro path** — operator flagged install instructions missing and "README still sucks massively"; this is the single highest-leverage GTM investment because it's the literal barrier between "impressive repo" and "a stranger actually ran it" — NEW
- **Invest in a rich, private-data-free demo corpus so every feature runs in 60s with zero user data** — the `demo seed`/`import --demo` path exists but needs a representative synthetic archive; removing the cold-start (no need to expose one's own sessions) is what converts a curious visitor into a runner — NEW
- **Top-of-README asset = a 90-second asciinema/animation of "cut a session mid-flight → agent recovers via recall"** — operator explicitly wants screencasts/images/badges/git-tags; a single visceral continuity demo out-persuades any prose feature list — NEW
- **Make construct-validity the *brand*, not just an internal invariant** — "every number is live-cited and cold-reader-verifiable" is a real differentiator in a field drowning in LLM slop dashboards; surface the rigor gate publicly as the promise, since it's the convergent theme prior agents already grounded — NEW
- **Permissive license (MIT/Apache), nothing held closed** — the moat is operator taste/execution, not code; copyleft or a closed core buys zero (no commercial competitor to fence out) and directly damages the legitimacy/citation payoff that is the actual point — NEW
- **Explicitly defer monetization; sustainability path is patronage/grant/employment leverage, not revenue** — raw-log frames the funding case as "a deal with parents" + resume/identity, not customers; declare this so no one builds signup funnels, and revisit "hosted convergence daemon for teams" only as a distant option — NEW
- **The funnel is docs, not signups: conversion = `git clone` + `import --demo` + read-through** — local-first single-writer means there's no account to create; GTM effort belongs in a task-oriented docs site (install → import → recall → findings), and the daemon/webui is a dogfood surface, not a SaaS onboarding path — NEW
- **Recruit 2-3 first-followers into the browser-capture + MCP recall dogfood loop for the only social proof that counts** — testimonials of the shape "my agents stopped re-asking / recovered on their own" are the category's currency; nothing else (stars, feature lists) substitutes for a peer confirming the unhobbling — NEW
- **Sequence the launch as a staircase, not a big bang** — (1) legibility epic + s7ae proof land → (2) install + demo corpus → (3) the data-story essay with live-cited findings → (4) one sharp thread into the AI-memory discourse; each step is independently shippable and each de-risks the next, matching the operator's "don't stall on undecidable design questions" directive — NEW

GPT-pro prompt stubs:
- **[DR]** "Survey the 2025-2026 landscape of local-first 'AI memory / agent context' tools (e.g. personal-knowledge + LLM-session substrates, agent-memory frameworks, context-engineering projects). For each: positioning, license, monetization model, distribution channel, first-user profile, and whether it succeeded. Identify the open positioning gap a single-writer, local, citation-rigorous session archive could credibly own, and the 2-3 discourse communities where its launch narrative would actually travel."
- **[A]** "Given a solo-built, local-first, OSS-by-necessity developer tool whose builder's real goals are (a) a fundable capability proof, (b) resume-gap legitimacy, (c) genuine personal utility — not revenue — design the optimal 'whole product' wrapper and launch sequence. Rank candidate investments (install ergonomics, demo corpus, screencast, docs site, essay, flagship live demo) by expected marginal effect on a *stranger's competence judgment* per unit effort, and state what to explicitly NOT build."
- **[A]** "Draft the canonical public narrative for a tool that lets AI coding agents read and recover from their own past sessions. Produce: a one-line category claim, a 3-sentence pitch, the single most persuasive 90-second demo script, and the top-of-README structure — optimized for an audience of technical agent-tooling practitioners, with construct-validity ('every number is live-cited and verifiable') as the differentiator against slop."