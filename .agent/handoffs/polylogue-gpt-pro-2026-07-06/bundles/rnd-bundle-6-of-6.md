# Polylogue R&D bundle — file 6 of 6

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
  PART 2 - PAYLOAD: 16 subagent outputs (unique to file 6 of 6)
=====================================================================


---

## [a13240c2de891b1d0] SPEC — Episode Unit (logical task spanning sessions/tools/time)

I have sufficient grounding. Here is the implementation-grade spec.

---

# SPEC — Episode Unit (logical task spanning sessions/tools/time)

**Scope boundary vs prior tiers.** `session_links` = within-provider, *identity-shared* lineage (fork/resume/subagent/acompact replay). `threads` = the connected-component grouping of that lineage (one root session). The **episode** is a tier *above* both: it groups *distinct* threads/sessions — possibly across providers, cwds, and days — by *thematic/temporal/artifact* evidence, never by shared identity. An episode is a hypothesis with confidence, not a fact read off a `parent_id`.

Grounding files: `polylogue/storage/sqlite/archive_tiers/index.py` (session_links:376, threads:410, repos/session_repos/session_commits:477-515, session_profiles:799), `.../embeddings.py` (vec0 float[1024], message-level only), `.../user.py` (assertions, v4), `core/enums.py` (AssertionKind:399, MaterialOrigin:176, LinkType:302), `core/refs.py` (ObjectRefKind:8), `insights/registry.py` (InsightType:70).

Critical pre-existing substrate I build **on**, not around: `repos`/`session_repos`/`session_commits` already exist (bead **7xv** partially landed — `session_commits.detection_type IN (time_window|file_overlap|explicit_ref|origin_reported)` with confidence+evidence_json is exactly the shared-artifact/commit-attribution substrate). `session_profiles` already carries `repo_paths_json`, `repo_names_json`, `first/last_message_at`, `canonical_session_date`, `total_cost_usd`, `per_model_cost_json`, `cost_provenance`, `terminal_state`, `logical_session_id` — this is the scorer's feature vector, no new capture needed.

---

## 1. Schema / DDL — tier & regime

Three placements, keyed by durability (matches `CLAUDE.md` schema-regime doctrine):

### 1a. `index.db` (DERIVED, rebuildable, no migration chain) — bump `INDEX_SCHEMA_VERSION 24 → 25`, edit canonical DDL + rebuild plan.

```sql
-- Episode: a stitched logical task. DERIVED — fully recomputed from source + user
-- judgments on rebuild. episode_id is DETERMINISTIC (anchor-keyed) so it survives
-- `ops reset --index && polylogued run`.
CREATE TABLE IF NOT EXISTS episodes (
    episode_id          TEXT PRIMARY KEY,     -- 'ep:' || anchor_session_id (see §2)
    anchor_session_id   TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    member_set_hash     BLOB NOT NULL CHECK(length(member_set_hash)=32), -- idempotency skip key
    member_count        INTEGER NOT NULL CHECK(member_count >= 1),
    confidence_tier     TEXT NOT NULL CHECK(confidence_tier IN ('linked','corroborated','candidate')),
    score               REAL NOT NULL CHECK(score BETWEEN 0 AND 1),
    dominant_repo_id    TEXT REFERENCES repos(repo_id) ON DELETE SET NULL,
    title               TEXT NOT NULL DEFAULT '',      -- derived: anchor title or LLM-free heuristic
    start_time          TEXT, end_time TEXT,
    span_ms             INTEGER NOT NULL DEFAULT 0 CHECK(span_ms >= 0),
    total_cost_usd      REAL NOT NULL DEFAULT 0.0,     -- lineage-deduped, §5
    total_credit_cost   REAL NOT NULL DEFAULT 0.0,
    cost_provenance     TEXT NOT NULL DEFAULT 'unknown',
    origins_json        TEXT NOT NULL DEFAULT '[]',    -- distinct Origin tokens spanned
    evidence_json       TEXT NOT NULL DEFAULT '{}',    -- per-signal contributions, auditable
    search_text         TEXT NOT NULL DEFAULT '',
    materializer_version INTEGER NOT NULL DEFAULT 1,
    materialized_at_ms  INTEGER NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS episode_members (
    episode_id     TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
    session_id     TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role           TEXT NOT NULL CHECK(role IN ('anchor','member','candidate')),
    edge_score     REAL NOT NULL CHECK(edge_score BETWEEN 0 AND 1),  -- best edge into the component
    evidence_json  TEXT NOT NULL DEFAULT '{}',        -- {repo,time,embed,artifact,sha[],errfp[]}
    PRIMARY KEY(episode_id, session_id)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_episode_members_session ON episode_members(session_id);

-- Scored stitch graph, persisted for audit + incremental replay. One row per candidate pair.
CREATE TABLE IF NOT EXISTS episode_edges (
    src_session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    dst_session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    score          REAL NOT NULL CHECK(score BETWEEN 0 AND 1),
    tier           TEXT NOT NULL CHECK(tier IN ('linked','corroborated','candidate','rejected')),
    floor_passed   INTEGER NOT NULL CHECK(floor_passed IN (0,1)),
    s_repo REAL NOT NULL, s_time REAL NOT NULL, s_embed REAL NOT NULL, s_artifact REAL NOT NULL,
    corroborators_json TEXT NOT NULL DEFAULT '[]',    -- ['sha','errfp','repo+time',...]
    negative_json      TEXT NOT NULL DEFAULT '[]',    -- anti-stitch reasons
    created_at_ms  INTEGER NOT NULL,
    PRIMARY KEY(src_session_id, dst_session_id)       -- src < dst canonical order
) STRICT;

-- episode → produced artifacts (commits/PRs). Built on existing session_commits + github refs.
CREATE TABLE IF NOT EXISTS episode_produced_refs (
    episode_id     TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
    ref_kind       TEXT NOT NULL CHECK(ref_kind IN ('commit','github-pr','github-issue','branch')),
    ref_value      TEXT NOT NULL,                     -- SHA / PR url / etc
    via_session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    confidence     REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    evidence_json  TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY(episode_id, ref_kind, ref_value)
) STRICT;
```

### 1b. `embeddings.db` (DERIVED, rebuildable) — bump `EMBEDDINGS_SCHEMA_VERSION 1 → 2`.

Embeddings today are **message-level only** (`message_embeddings`). The scorer's "embedding cosine over session summaries" needs a **session-level vector** that does not exist yet:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS session_embeddings USING vec0(
    session_id TEXT PRIMARY KEY,
    embedding  float[1024],
    +origin TEXT
);
CREATE TABLE IF NOT EXISTS session_embeddings_meta (
    session_id     TEXT PRIMARY KEY,
    method         TEXT NOT NULL CHECK(method IN ('summary_text','message_centroid')),
    content_hash   BLOB NOT NULL CHECK(length(content_hash)=32),  -- of source summary/messages
    model          TEXT NOT NULL,
    embedded_at_ms INTEGER,
    needs_reindex  INTEGER NOT NULL DEFAULT 0 CHECK(needs_reindex IN (0,1))
) STRICT;
```

Two methods, decided at build: `message_centroid` (mean-pool of existing `message_embeddings` filtered to `human_authored|assistant_authored` — zero extra Voyage calls, reuses the message vectors) as the **default/cheap path**; `summary_text` (embed `session_profiles.search_text`/title) as a fallback when a session has no embeddable messages. Centroid is preferred because it's free and already gated by `material_origin` via `EMBEDDABLE_MESSAGE_WHERE`.

### 1c. `user.db` (DURABLE) — **no schema bump needed**.

Durable stitch judgments are `assertions` rows. `kind` is plain `TEXT` (vocabulary grows without a user-tier migration). Add three closed **runtime** values to `AssertionKind` (`core/enums.py`) — `EPISODE_CONFIRM`, `EPISODE_SPLIT`, `EPISODE_REJECT` — and add `"episode"` to the `ObjectRefKind` Literal (`core/refs.py`). Judgments are **edge-keyed**, not episode-keyed, so they survive re-materialization:

- `EPISODE_REJECT`: `target_ref = "session:<A>"`, `value_json = {"pair":["<A>","<B>"]}` — a permanent negative edge (anti-stitch).
- `EPISODE_CONFIRM`: same pair shape — pins an edge to `linked` tier regardless of score.
- `EPISODE_SPLIT`: `value_json = {"cut":[["<A>","<B>"],...]}` — forbidden-merge cut constraint.

> Gotcha (from project memory): a new `AssertionKind`/`ObjectRefKind` is DB-schema-free but is embedded in `render openapi` + `render cli-output-schemas`; regenerate or `render all --check` fails (grep `out of sync`, don't trust the tail line). Adding modules also requires `devtools render topology-projection && topology-status`.

---

## 2. Stitch algorithm + confidence model (pseudocode)

```python
# ---- Candidate generation (blocking) — never score all N² (8.8K logical sessions = 38M pairs) ----
def candidate_pairs(sessions):
    seen = set()
    for a in sessions:
        # Block 1: shared repo_id (session_repos)          — strong prior
        # Block 2: temporal window ±T days (session_profiles.canonical_session_date)
        # Block 3: embedding ANN — vec0 KNN top-k over session_embeddings (thematic, cross-repo)
        for b in (by_repo(a) | by_time(a, T=7d) | by_embed_knn(a, k=20)):
            if a.id < b.id and (a.id,b.id) not in seen:
                seen.add((a.id,b.id)); yield (a,b)

# ---- Four-signal scorer (pure function; unit-tested against a false-merge corpus) ----
def score_edge(a, b, judgments):
    # negative evidence FIRST — hard veto
    if reject_asserted(a,b, judgments):                    return Edge(score=0, tier='rejected', neg=['user_reject'])
    neg = negative_evidence(a,b)   # e.g. disjoint non-empty repos w/ no shared artifact = weak negative
    # signals ∈ [0,1]
    s_repo = repo_signal(a,b)          # 1.0 same repo_id; partial shared-cwd-subpath; 0 if both set & disjoint
    s_time = exp(-dt(a,b) / tau(a,b))  # repo-CONDITIONED kernel: tau tight when same active repo, wide cross-repo
    s_embed = max(0, cosine(vec(a), vec(b)))               # session_embeddings
    sha  = a.commit_shas & b.commit_shas                    # session_commits
    errfp = a.error_fingerprints & b.error_fingerprints     # normalized tool_result error text, blocks.tool_result_is_error
    files = jaccard(a.file_paths, b.file_paths)             # tool blocks + session_working_dirs
    s_artifact = weighted_max(sha_overlap=len(sha), err_overlap=len(errfp), file_jaccard=files)

    raw = W_REPO*s_repo + W_TIME*s_time + W_EMBED*s_embed + W_ART*s_artifact   # Σw=1

    # ---- HARD FALSE-MERGE FLOOR: no 'linked'/'corroborated' without a STRONG CORROBORATOR ----
    corroborators = []
    if sha:                              corroborators.append('sha')      # near-decisive
    if errfp:                            corroborators.append('errfp')
    if s_repo == 1 and s_time > THETA_T: corroborators.append('repo+time')
    if s_embed > THETA_E and s_time > THETA_T: corroborators.append('embed+time')
    floor_passed = len(corroborators) >= 1

    if confirm_asserted(a,b, judgments): return Edge(1.0, 'linked', corroborators or ['user_confirm'], floor=1)
    if neg:                              raw *= NEG_PENALTY
    if not floor_passed:
        # thematic-only similarity (e.g. two repos both "fix auth") is capped at 'candidate' — cannot merge
        return Edge(min(raw, THETA_CORR - eps), 'candidate', corroborators, floor=0, neg=neg)
    tier = ('linked'       if raw >= THETA_LINK and len(corroborators) >= 2
            else 'corroborated' if raw >= THETA_CORR
            else 'candidate')
    return Edge(raw, tier, corroborators, floor=1, neg=neg)

# ---- Episode construction: connected components over corroborated+ edges; SPLIT cuts; candidates attach loose ----
def materialize_episodes(sessions, judgments):
    edges = [score_edge(a,b,judgments) for a,b in candidate_pairs(sessions)]
    G = graph(nodes=sessions,
              edges=[e for e in edges if e.tier in ('linked','corroborated')
                     and not split_cuts(e, judgments)])          # SPLIT = forbidden-merge constraint
    for comp in connected_components(G):
        anchor = min(comp, key=lambda s: (s.first_message_at, s.session_id))
        eid = f"ep:{anchor.session_id}"                          # DETERMINISTIC, survives rebuild
        h = sha256(sorted(s.session_id for s in comp))           # member_set_hash — idempotency skip
        if episodes.get(eid).member_set_hash == h: continue       # unchanged → skip write
        tier = worst_edge_tier(comp)                              # component tier = min over member edges
        write_episode(eid, anchor, comp, h, tier, cost=rollup_cost(comp))  # §5
        for s in candidate_neighbors(comp, edges):                # attach candidate-tier edges as role='candidate'
            write_member(eid, s, role='candidate')
```

**Confidence tiers.** `linked` = user-confirmed OR score≥θ_link with ≥2 independent strong corroborators (e.g. shared SHA + same repo). `corroborated` = ≥1 strong corroborator, score≥θ_corr. `candidate` = weak thematic/temporal only (floor NOT passed) — surfaced but **never merged into a linked episode**; attached as `role='candidate'` for human review. Only `corroborated+` edges build the connected component. Thresholds live in one tuned constants module (`W_*`, `THETA_*`, `tau`), calibrated against a labeled fixture corpus — never hard-coded across call sites.

---

## 3. Rebuild plan

Derived-tier doctrine: no upgrade helper. Two version bumps ride the same rebuild:

1. Edit canonical DDL (`index.py` v25, `embeddings.py` v2) + the rebuild-plan doc entry. `devtools lab policy schema-versioning` must stay green (rejects an in-place upgrade path).
2. `polylogue ops reset --index --embeddings && polylogued run` rebuilds both derived tiers from `source.db`. `user.db` judgments are untouched (durable) and **replayed** during episode materialization — a reset then rebuild must reproduce identical `episode_id`s and re-apply every REJECT/CONFIRM/SPLIT.
3. Wire episode materialization as a **`DaemonConverger` stage** (`daemon/convergence_stages.py`) after `session_profiles` + `session_embeddings` are current. Use `false_means_pending`: bound work per pass (e.g. N newest sessions' candidate neighborhoods), push the remaining backlog to `convergence_debt` — episodes are "deferred until quiet," never block ingest. Session-embedding centroid build is an embeddings catch-up stage in the same converger (reuses the message-embedding catch-up machinery).
4. Regenerate: `render openapi`, `render cli-output-schemas` (new AssertionKind/ObjectRef), `render topology-projection`+`topology-status` (new modules), `render cli-reference`/`mcp-reference`. Verify `render all --check` (grep `out of sync`).

---

## 4. Test strategy (incl. false-merge)

- **Idempotent materialization** (property): same source → identical `episode_id` + stable `member_set_hash`; second run is a no-op skip.
- **Durable key survives rebuild** (integration): seed episode, assert `EPISODE_REJECT`, `ops reset --index`, rebuild → episode re-materializes and the reject still cuts the edge.
- **False-merge floor** (the load-bearing suite — construct validity): two sessions in **different repos**, same day, high embedding cosine (both literally "fix the auth bug" in unrelated projects) → must resolve to `tier='candidate'` at most, **never** `linked`/`corroborated`. Parameterize: high-embed+disjoint-repo, high-embed+no-artifact, same-title-different-cwd. Assert component does NOT merge.
- **Decisive corroborator**: two sessions across *different providers* (claude-code + codex) touching the **same commit SHA** → `linked`. Same for shared error fingerprint.
- **Anti-stitch negative evidence**: pair the scorer *would* link, plus `EPISODE_REJECT` → edge excluded; pair with weak negative (disjoint repos) gets `NEG_PENALTY` and can't reach `linked`.
- **SPLIT constraint**: a confirmed 3-session component + `EPISODE_SPLIT` cut → two episodes; anchors re-derived correctly.
- **CONFIRM promotion**: a below-threshold candidate + `EPISODE_CONFIRM` → `linked`, pinned.
- **Confidence monotonicity** (property): adding a corroborator to an edge never lowers its tier; removing negative evidence never lowers score.
- **Lineage-dedup cost** (against #2467): episode containing a fork/resume member must NOT double-count the replayed prefix cost — sum over logical sessions, tail-only for prefix-sharing children (assert vs a hand-computed fixture).
- **Blocking soundness**: a known same-task cross-provider pair with shared SHA but >7d apart and different repo *must* still be a candidate pair via the ANN block (guards against blocking dropping true positives).
- Clock hygiene: use `frozen_clock`; temporal-kernel tests must not touch wall clock (`verify-test-clock-hygiene` lint). Use `SessionBuilder`/`corpus_seeded_db`.

---

## 5. Episode cost rollup (honoring material_origin + lineage-dedup)

`total_cost_usd` = Σ over the episode's **logical sessions**, deduping lineage replay: for a `session_links` child with `inheritance='prefix-sharing'`, count only the divergent-tail cost (the parent prefix's cost belongs to the parent) — resolve via `branch_point_message_id`. `spawned-fresh` subagents count fully. Authored-token / user-word rollups exclude `material_origin IN (runtime_protocol, runtime_context, tool_result, operator_command)` (same filter as `EMBEDDABLE_MESSAGE_WHERE`); dollar cost includes all. Carry `cost_provenance` = worst-of members (`unknown` dominates). Reuse the existing per-session cost readers (`session_profiles.total_cost_usd`/`per_model_cost_json`) — do not re-derive from blocks.

---

## 6. Bead breakdown (acceptance criteria)

Suggested parent epic sibling to **7xv** (native git/repo awareness — episodes consume its `session_commits`). Not created (spec-only).

1. **Session-level embeddings (embeddings.db v2).** `session_embeddings` vec0 + meta; `message_centroid` default (reuse message vectors, `material_origin`-gated), `summary_text` fallback; converger catch-up stage. *AC:* every eligible session has a vector; centroid reuses existing message embeddings (0 new Voyage calls for sessions with embeddable messages); `embedding_status`-style pending/stale tracking green.

2. **Four-signal scorer core (pure).** Candidate blocking (repo/time/ANN) + `score_edge` + floor + negative evidence + tiers, in one constants-tunable module. *AC:* false-merge corpus (§4) passes — disjoint-repo high-embed pairs never exceed `candidate`; shared-SHA cross-provider pairs reach `linked`; scorer is a pure function with no DB coupling (unit-tested in isolation).

3. **Episode tables + materializer (index.db v25).** DDL, connected-components over corroborated+ edges, anchor-keyed deterministic `episode_id`, `member_set_hash` idempotency, `episode_edges` audit. *AC:* idempotent rebuild reproduces identical ids; materializer wired as a `false_means_pending` converger stage; `schema-versioning` policy green.

4. **Durable stitch judgments.** `AssertionKind.EPISODE_{CONFIRM,SPLIT,REJECT}`, `ObjectRefKind "episode"`, edge-keyed targeting, materializer replay. Regenerate openapi + cli-output-schemas. *AC:* reject/confirm/split survive `ops reset --index`; `render all --check` clean; user_audit surfaces the new kinds.

5. **Episode→produced-ref attribution + cost rollup.** `episode_produced_refs` from `session_commits` + github refs; lineage-deduped `material_origin`-honoring cost. *AC:* episode's commits/PRs union its members' attributed refs above confidence; fork-member episode cost equals hand-computed tail-only sum (#2467 regression fixture).

6. **Read surfaces.** `InsightType` descriptor (`episode`), DSL `episode:` predicate + `analyze episodes` projection, MCP `get_episode`/`list_episodes`/`confirm_episode`/`split_episode`/`reject_stitch` (update `EXPECTED_TOOL_NAMES` + contracts). *AC:* CLI/JSON/MCP all render from the one descriptor; origin projection applied at boundary; discovery tests green.

7. *(optional, off by default)* **Lynchpin/Sinex cross-source commit corroboration.** Strengthen episode→commit edges with external commit authorship/timestamp evidence via the Lynchpin MCP; bounded, opt-in. *AC:* enrichment only raises confidence with recorded evidence_refs, never fabricates edges; disabled → episodes still materialize from in-archive evidence alone.

---

## 7. Top-3 risks

1. **False merge / construct validity (highest).** Embedding cosine conflates *thematic* similarity with *same task* — two unrelated repos both "fixing auth" will score high on embed+time. This is precisely the project's claim-vs-evidence / construct-validity doctrine (memory: "fix what data implicates," `insight_rigor_audit`). Mitigation is structural, not tuned: the hard floor requires a **strong structural corroborator** (shared SHA / error fingerprint / same-repo+time) before any `corroborated+` tier; disjoint-repo is negative evidence; thematic-only stays `candidate`. If this floor is ever relaxed to boost recall, episodes become fiction. The false-merge suite (§4) is the guard and must be treated as protected.

2. **Durable-vs-derived key stability.** Episodes are derived (rebuildable, must re-key from source) but user judgments are durable — a naive whole-set or content-hash episode id would orphan every confirm/reject on the next `ops reset --index`. Mitigation: anchor-derived deterministic id + **edge-keyed** (session-pair) judgments that replay independent of episode identity. Residual: an earlier-timestamped session backfilled into a component re-anchors the episode id — tolerable (judgments are pair-keyed, unaffected) but the rebuild test must assert it.

3. **Scale + lineage-dedup correctness.** N² over ~8.8K logical sessions is 38M pairs; blocking must be both cheap and *complete* (a dropped true pair is an invisible miss). ANN block depends on session embeddings existing (bead-1 gates bead-2/3). Separately, cost rollup inherits #2467: `session_profiles` cost for prefix-sharing children may already be tail-only or double-counted depending on how the profile was materialized — the rollup must dedup via `session_links`, and the regression fixture (bead-5 AC) is the only thing that catches a silent 2× cost inflation at the episode level.

---

## [a29d7a5a4e98bd550] Wave-5 Implementation Spec — three missing units

I have complete grounding. Writing the spec now.

---

# Wave-5 Implementation Spec — three missing units

Grounded in: `polylogue/storage/sqlite/archive_tiers/index.py` (index.db DDL, `INDEX_SCHEMA_VERSION = 24`), the `actions` VIEW (index.py:324–343), `session_observed_events` (index.py:952), `session_commits` (index.py:502), `session_runs` (index.py:913), `query_files` (archive.py:5115), the query-unit registry (`archive/query/metadata.py`: `QueryUnitName`, `QUERY_UNIT_DESCRIPTORS`, `StructuralQueryUnitInfo`), structural ref extraction (`insights/transforms.py`: `_PR_RE`/`_ISSUE_RE`/`_COMMIT_SHA_RE`, `ToolSummary.{pr,issue,file,commit}_refs`), `ObservedEventKind` (`insights/run_projection.py:22`), and the missing-units analysis (`.agent/scratch/rnd-brainstorm-2026-07-05.md`, W2-U line 502, themes 3/4/9/11).

## Table-vs-VIEW decision matrix (the A28 axes, made operational)

| Axis | VIEW when… | TABLE when… |
|---|---|---|
| **Derivation cost / recursion depth** | bounded self-join or 1–2 joins over already-materialized rows (the `actions` precedent: one self-join on `tool_id`) | multi-pass scoring, **regex/JSON scan over `blocks.search_text`**, or cross-session aggregation |
| **Query-time join cost** | match keys are indexed columns; re-executing per query is cheap at 16k sessions | correlation needs a scan the planner can't index; per-query re-derivation is prohibitive |
| **Rebuild vs query economics** (index.db is rebuildable) | cheap to recompute, rarely queried | expensive to derive **once**, frequently queried → pay at materialize, not per read |
| **Query-unit exposure** | internal join helper, or a unit whose columns are all stable projections of underlying indexed columns | first-class DSL unit needing its own confidence/candidate axis, dedicated indexes, or a taxonomy column that doesn't exist upstream |

**Verdicts:** `entity-mention` → **TABLE** (regex extraction is not re-runnable per query; needs a structural-vs-candidate trust axis). `world-effect` → **VIEW** (unnest `evidence_refs_json` → tool_use block; bounded, indexed). `verification-run` → **VIEW** (projection over `actions` ⋈ the already-materialized verification `observed_events`; exit_code is the keystone, no new derivation).

All three live in **`index.db` (derived tier)** → the migration path is **edit canonical DDL + bump `INDEX_SCHEMA_VERSION` + rebuild** (`polylogue ops reset --index && polylogued run`). **No numbered SQL migration** (those are for `source.db`/`user.db` only; `devtools lab policy schema-versioning` rejects derived-tier upgrade helpers).

---

## Unit 1 — `entity-mention` (TABLE)

A mention edge: session/message/block → a referenced entity (file path, `#N` issue, PR, URL, code symbol, commit SHA). Structural mentions (from `tool_input`/`tool_result` fields, `web_content_constructs.url`, `session_commits.commit_sha`) are trusted; prose-mined mentions (regex over `blocks.search_text`) are `candidate` and recursive-safety-gated.

### (1) Schema / DDL — `index.db`, TABLE

```sql
CREATE TABLE IF NOT EXISTS entity_mentions (
    mention_ref     TEXT GENERATED ALWAYS AS
                        (block_id || ':mention:' || position) STORED UNIQUE,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    message_id      TEXT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
    block_id        TEXT NOT NULL REFERENCES blocks(block_id) ON DELETE CASCADE,
    position        INTEGER NOT NULL CHECK(position >= 0),
    -- closed vocabulary; embedded from a Python Literal via literal_check()
    entity_kind     TEXT NOT NULL CHECK(entity_kind IN
                        ('file','issue','pr','url','symbol','commit_sha')),
    entity_value    TEXT NOT NULL,                 -- normalized: path fwd-slashed, sha lowered, #N->issue number
    entity_key      TEXT NOT NULL,                 -- canonical dedup key (kind || ':' || value)
    -- structural = read from a typed field (tool_input.file_path, web_content_constructs.url,
    -- session_commits); candidate = regex-mined from search_text (prose). Mirrors
    -- session_commits.detection_type + confidence trust pattern.
    provenance      TEXT NOT NULL CHECK(provenance IN ('structural','candidate')),
    detection_type  TEXT NOT NULL CHECK(detection_type IN
                        ('tool_input','tool_result','web_construct','commit_row','prose_regex')),
    confidence      REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    evidence_json   TEXT NOT NULL DEFAULT '{}',
    materializer_version INTEGER NOT NULL DEFAULT {SESSION_INSIGHT_MATERIALIZER_VERSION},
    materialized_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(block_id, position)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_entity_mentions_key
ON entity_mentions(entity_kind, entity_key);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_session
ON entity_mentions(session_id, position);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_value
ON entity_mentions(entity_value);            -- powers "which sessions mention file X / PR #N"
```

Register `entity_mentions` in the `insight_materialization.insight_type` CHECK list (index.py:675) and add `EntityMentionKind` Literal to `core/enums.py` so `literal_check` stays in lockstep.

### (2) Derivation algorithm

```
materialize_entity_mentions(session):
  for block in session.blocks:
    # --- STRUCTURAL (confidence 1.0, provenance=structural) ---
    if block.block_type == 'tool_use':
        for field in ('file_path','path','notebook_path'):
            p = json_extract(tool_input, field)
            if p: emit(kind='file', value=normalize_path(p),
                       detection_type='tool_input', confidence=1.0, provenance='structural')
    if block.block_type == 'tool_result':
        # reuse ToolSummary refs already parsed in transforms.py from typed structure
        for sha in tool.commit_refs:   emit('commit_sha', lower(sha), 'tool_result', 1.0, structural)
        for pr  in tool.pr_refs:       emit('pr',   pr,  'tool_result', 1.0, structural)
        for iss in tool.issue_refs:    emit('issue',iss, 'tool_result', 1.0, structural)
    # web_content_constructs already parse URLs/queries structurally:
  for wc in session.web_content_constructs where wc.url:
        emit('url', wc.url, 'web_construct', 1.0, structural)
  for c in session.session_commits:
        emit('commit_sha', c.commit_sha, 'commit_row', c.confidence, structural)

    # --- CANDIDATE (prose-mined; recursive-safety gated) ---
    # only over human/assistant material_origin blocks; NEVER generated_context_pack / tool_result prose
    if block.material_origin in {human_authored, assistant_authored}:
        for m in (_PR_RE | _ISSUE_RE | _COMMIT_SHA_RE | _URL_RE | _SYMBOL_RE).finditer(block.text):
            key = (kind, normalize(m))
            if key already emitted structurally for this block: skip   # structural wins
            emit(kind, value, 'prose_regex', confidence=0.4, provenance='candidate')

  dedup by (block_id, position); structural rows suppress candidate rows with same entity_key in same block.
```

Runs as a `ConvergenceStage`/materializer (alongside `observed_events`), CPU-bound → `ProcessPoolExecutor`; main process is sole writer. Idempotent per session (delete-by-session then re-insert, the existing insight pattern).

### (3) Migration
Derived tier: add DDL block to `INDEX_DDL`, bump `INDEX_SCHEMA_VERSION` 24→25, add rebuild note. `polylogue ops reset --index && polylogued run` repopulates. Add `entity_mentions` to `insight_materialization` CHECK. No numbered migration file.

### (4) Test strategy
- **Property**: every `structural` row's `entity_value` is byte-present in the cited block's typed field (no fabrication) — the "resolves to bytes" invariant.
- **Property**: no `candidate` row over a `generated_context_pack`/`tool_result` block (recursive-safety / laundering gate).
- **Unit**: `_SYMBOL_RE`/`_URL_RE` extraction fixtures; structural-suppresses-candidate dedup; path normalization (backslash→slash, matching `query_files` REPLACE).
- **DSL**: `entity-mentions where kind:pr | group by session.repo | count` and `exists entity-mention(kind:file AND value:archive/query)` round-trip through parser→SQL.
- **Regression**: the recovery-digest fabrication class (MEMORY.md) — a mined PR number renders as `candidate`, never as fact.

### (5) Bead
- **title:** `feat(units): entity-mention table — structural vs prose-mined reference edges`
- **type:** feature, epic-anchor **W2-U** / links themes #3 (citation anchors), #9 (recursive-safety)
- **AC:** (a) `entity_mentions` table + indexes in index.db, `INDEX_SCHEMA_VERSION` bumped; (b) materializer emits structural rows from `tool_input`/`tool_result`/`web_content_constructs`/`session_commits`, candidate rows regex-gated to authored material_origin; (c) `entity-mention` registered as a query unit (`QueryUnitName`, `QUERY_UNIT_DESCRIPTORS`, `_entity_mention_field_infos`, `query_entity_mentions`, `EntityMentionQueryRowPayload`) with `provenance`/`kind` in `aggregate_group_fields`; (d) `render cli-output-schemas` + `render openapi` + topology projection regenerated; (e) property tests for byte-presence + no-prose-over-generated-context pass.

### (6) Top-3 risks
1. **Prose-regex false positives** (7–40-hex `_COMMIT_SHA_RE` matches hashes/IDs everywhere) → cap candidate confidence at 0.4, require ≥7 chars + word boundary, keep them a separate `candidate` provenance so they never leak as fact.
2. **Symbol extraction has no structural source** (no typed field yields code symbols) → `symbol` is candidate-only at v1; defer a tree-sitter-backed structural symbol pass to a follow-up rather than shipping low-trust symbols as structural.
3. **Table growth** (mentions are per-block, could be 5–10× messages) → indexes chosen for the two real queries (by entity_key, by session); measure row count on rebuild before exposing an unfiltered `entity-mentions | rows` terminal.

---

## Unit 2 — `world-effect` (VIEW)

A cause→effect edge: an `observed_event` (a commit landed / file written / test run) linked to the **action/turn that caused it**. Grounded in the fact that `session_observed_events` already carries `evidence_refs_json` (pointing at the producing tool_use block) and `object_refs_json` (the affected entity), materialized structurally.

### (1) Schema / DDL — `index.db`, VIEW

```sql
CREATE VIEW IF NOT EXISTS world_effects AS
SELECT
    e.event_ref,
    e.session_id,
    e.kind                    AS effect_kind,     -- command_succeeded/failed, test_passed/failed, tool_finished
    e.delivery_state,
    -- cause side: unnest the first evidence ref (a block_id) -> the causing action
    a.tool_use_block_id       AS cause_action_block_id,
    a.tool_name               AS cause_tool,
    a.tool_command            AS cause_command,
    a.tool_path               AS cause_path,
    a.is_error                AS cause_is_error,
    a.exit_code               AS cause_exit_code,
    e.subject_ref,
    e.object_refs_json        AS effect_object_refs,
    e.position,
    e.summary
FROM session_observed_events e
LEFT JOIN json_each(e.evidence_refs_json) ev ON ev.key = 0        -- primary evidence ref
LEFT JOIN blocks cb ON cb.block_id = json_extract(ev.value, '$.block_id')
LEFT JOIN actions a
       ON a.session_id = e.session_id
      AND a.tool_use_block_id = cb.block_id;
```

The join is on `evidence_refs_json[0].block_id` → `actions.tool_use_block_id` (both indexed: `idx_blocks_tool_id`, `idx_session_observed_events_run`). This is **structural attribution**, not temporal guessing — the observed-event materializer already recorded which tool_use produced it. A temporal/path fallback (join `session_commits` by time-window when no evidence ref exists) is available via a second `UNION ALL` branch but is `detection_type='time_window'`-tier confidence and should be gated behind an explicit lower-trust flag.

### (2) Derivation algorithm
Pure SQL VIEW — no materializer. Effect rows come from the existing `observed_events` materializer (which already runs the `_PR_RE`/`_COMMIT_SHA_RE`/test-count structural extraction in `transforms.py` and stores `evidence_refs_json`). The VIEW only **joins cause↔effect on read**. For the temporal-fallback branch (world-effect where a commit has no in-session evidence ref — the "terminal-artifact glue" from W2-Ep line 499): correlate `session_commits` ⋈ nearest preceding action by `created_at_ms`/`tool_path` overlap, emitted as `detection_type='time_window'`.

### (3) Migration
Add VIEW to `INDEX_DDL`, bump `INDEX_SCHEMA_VERSION`. VIEWs re-create on every open (`CREATE VIEW IF NOT EXISTS`), so rebuild is free. No new table, no `insight_materialization` entry.

### (4) Test strategy
- **Contract**: every `world_effects` row with a non-null `cause_action_block_id` resolves that block to a real `tool_use` block in the same session (referential integrity across the JSON-unnest join).
- **Semantic**: a `test_failed` observed-event links to the pytest/verify action whose `is_error=1`/`exit_code!=0` — the keystone consistency check (v16 fields agree with the event kind).
- **DSL**: `world-effects where effect_kind:command_failed | group by cause_tool | count`.
- **Null-safety**: observed-events with empty `evidence_refs_json` yield a row with null cause (not dropped) — degradation-honesty (brainstorm F: EVIDENCE-ONLY floor).

### (5) Bead
- **title:** `feat(units): world-effect VIEW — cause→effect edge from observed-event to producing action`
- **type:** feature, links **W2-U** line 502, **W2-Ep** attribution (line 499), theme #4 ("so-what is one materializer away")
- **AC:** (a) `world_effects` VIEW joining `session_observed_events.evidence_refs_json` → `actions`; (b) registered as query unit `world-effect` with `effect_kind`/`cause_tool`/`cause_exit_code` group fields; (c) keystone-consistency test (event kind agrees with cause exit_code/is_error); (d) null-cause rows preserved; (e) optional time-window fallback branch is confidence-tagged and off by default; (f) generated surfaces + topology regenerated.

### (6) Top-3 risks
1. **`evidence_refs_json[0]` assumption** — if the primary evidence ref isn't always the causing tool_use (could be a message ref), the join silently drops. → Validate the ref-kind in the VIEW (`json_extract(...,'$.kind')='block'`) and add a materializer contract that observed-events set evidence[0] = producing block.
2. **`json_each` join cost per query** — VIEW unnests JSON on every read. At 16k sessions this is bounded by observed-event count but should be measured; if the `EXPLAIN` shows no index use, precompute `cause_action_block_id` as a stored column on `session_observed_events` (moves it toward table).
3. **Temporal-fallback false attribution** — the W2-Ep line 497 "hard false-merge floor" applies: a commit correlated by time-window to the wrong action fabricates causality. → keep the fallback branch confidence-tagged, off by default, never in the structural VIEW's default projection.

---

## Unit 3 — `verification-run` (VIEW, extends `actions`/`runs`)

A check/pytest/verify/build invocation with pass/fail derived from the keystone `is_error`/`exit_code`. The verification-outcome observed-events (`test_passed`/`test_failed`/`command_succeeded`/`command_failed`) **already exist** in `ObservedEventKind` and are materialized — so this unit is a projection, not new derivation (brainstorm theme #10: "wire what already exists").

### (1) Schema / DDL — `index.db`, VIEW over `actions` ⋈ verification `observed_events`

```sql
CREATE VIEW IF NOT EXISTS verification_runs AS
SELECT
    a.session_id,
    a.message_id,
    a.tool_use_block_id       AS action_block_id,
    a.tool_name,
    a.tool_command,
    a.tool_path,
    -- command classifier: pytest / verify / build / typecheck / lint / test-generic
    CASE
      WHEN a.tool_command LIKE '%pytest%' OR a.tool_command LIKE '%devtools test%' THEN 'pytest'
      WHEN a.tool_command LIKE '%devtools verify%'                                 THEN 'verify'
      WHEN a.tool_command LIKE '%mypy%'                                            THEN 'typecheck'
      WHEN a.tool_command LIKE '%ruff%' OR a.tool_command LIKE '%lint%'            THEN 'lint'
      WHEN a.tool_command LIKE '%cargo test%' OR a.tool_command LIKE '%test%'      THEN 'test'
      WHEN a.tool_command LIKE '%build%' OR a.tool_command LIKE '%nix build%'      THEN 'build'
      ELSE NULL
    END                       AS verification_kind,
    -- outcome is STRUCTURAL from the v16 keystone, never regex over prose:
    CASE
      WHEN a.exit_code IS NOT NULL THEN (a.exit_code = 0)
      WHEN a.is_error  IS NOT NULL THEN (a.is_error = 0)
      ELSE NULL                                            -- unknown, honest NULL
    END                       AS passed,
    a.exit_code,
    a.is_error,
    e.event_ref               AS outcome_event_ref         -- the materialized test_passed/failed event, when present
FROM actions a
LEFT JOIN session_observed_events e
       ON e.session_id = a.session_id
      AND json_extract(e.evidence_refs_json, '$[0].block_id') = a.tool_use_block_id
      AND e.kind IN ('test_passed','test_failed','command_succeeded','command_failed')
WHERE (verification_kind) IS NOT NULL;
```

Outcome resolution order: `exit_code` (most authoritative) → `is_error` → `NULL` (unknown, never guessed). This is exactly the keystone-index rationale (index.py:217–224). It **extends `runs`** by keying on the same `session_id`/action lineage; a `run_ref` join is available for subagent-scoped rollups.

### (2) Derivation algorithm
Pure SQL VIEW. Classification is a bounded `LIKE`/CASE over `tool_command` (short strings, not full text). No materializer, no regex over `search_text`. If the command taxonomy needs to grow beyond CASE-legibility, promote the classifier to a small `verification_command_patterns` config table joined in — but v1 is inline CASE (the same shape as `actions.semantic_type`).

### (3) Migration
Add VIEW to `INDEX_DDL`, bump `INDEX_SCHEMA_VERSION`. Free rebuild. No table.

### (4) Test strategy
- **Keystone property**: `passed` is `TRUE` iff `exit_code=0` (or is_error=0 when exit_code null), `NULL` when both unknown — never inferred from output prose. This is the anti-fabrication core.
- **Classifier fixtures**: `devtools verify`, `pytest -k x`, `mypy --strict`, `cargo test`, `nix build` each map to the right `verification_kind`; a `git status` does **not** appear (not a verification).
- **Consistency**: when an `outcome_event_ref` is present, its kind (`test_failed`) agrees with `passed=FALSE`.
- **DSL**: `verification-runs where verification_kind:pytest AND passed:false | group by session.repo | count` (the "wasted-test-run / retry-storm" measures, brainstorm B).
- **Coverage-gate**: an insight built on this (pass-rate) over zero verification rows must FAIL, not render 0% (theme #12).

### (5) Bead
- **title:** `feat(units): verification-run VIEW — pass/fail from exit_code keystone over actions`
- **type:** feature, links **W2-U** line 503, theme #4/#10, feeds measures 9l5.x (retry-storm, wasted-test-run)
- **AC:** (a) `verification_runs` VIEW classifying verification actions + resolving outcome from `exit_code`→`is_error`→NULL; (b) registered query unit `verification-run` with `verification_kind`/`passed`/`session.repo` group fields; (c) keystone property test (no prose inference) + classifier fixtures; (d) zero-row insight fails the coverage gate; (e) `outcome_event_ref` join to existing verification observed-events verified consistent; (f) generated surfaces + topology regenerated.

### (6) Top-3 risks
1. **Command classifier brittleness** — `LIKE '%test%'` over-matches (`latest`, `test_helper.py` reads); a project's custom verify alias is missed. → anchor patterns to command-leading tokens where possible, keep taxonomy in one place, and treat `verification_kind` as best-effort while `passed` (from exit_code) stays authoritative.
2. **`exit_code`/`is_error` both NULL for web/Claude-web origins** (no structured tool result) → `passed IS NULL` dominates for non-CLI origins; document this and scope pass-rate measures to origins that populate the keystone (claude-code, codex).
3. **Double-counting vs `observed_events`** — both this VIEW and the `test_passed/failed` observed-events describe the same outcome; a naive union in downstream measures double-counts. → this VIEW is the **action-grained** truth; observed-events are the **event-grained** projection; measures must pick one frame (the denominator/unit-frame construct-validity trap, brainstorm W2-M line 450).

---

## Cross-cutting wiring (applies to all three)

Adding any query unit touches this exact surface set (from the registry I read): `QueryUnitName` Literal + `QUERY_UNIT_DESCRIPTORS` + a `StructuralQueryUnitInfo` + `_<unit>_field_infos()` + a `query_<unit>` method on `ArchiveStore` (archive.py, alongside `query_files`/`query_observed_events`) + a `<Unit>QueryRowPayload` model. Consumers that must not drift: `cli/shell_completion_values.py`, `mcp/server_tools.py` (+ `EXPECTED_TOOL_NAMES` if a dedicated MCP tool), `daemon/http.py` (`/api/query-units`). New module/table → regenerate topology projection or `render all --check` fails; new enums → regenerate `render cli-output-schemas` + `render openapi`. Verify per-unit with `devtools test <files>` (not blanket), then `devtools verify` once before PR.

**Sequencing recommendation:** `entity-mention` first (it's the only one needing a materializer + rebuild; land its `INDEX_SCHEMA_VERSION` bump and batch the two VIEW additions into the *same* bump to avoid two live rebuilds of the 38GB archive). `world-effect` and `verification-run` are VIEW-only and can ride that single schema bump.

---

## [afa6f19e7e7655c08] gjg — Compaction Lifecycle: Snapshot / Forensics / Re-grounding — IMPLEMENTATION SPEC

I have sufficient grounding. Here is the implementation-grade spec.

---

# gjg — Compaction Lifecycle: Snapshot / Forensics / Re-grounding — IMPLEMENTATION SPEC

Grounded in: bead `polylogue-gjg` (epic) + deps `4ts.5` (boundary ranges / `get_effective_context`), `d1y` (hook install/liveness); `9l5.7` (measure registry); `37t.11` (ContextSource scheduler); `session_links`/`session_events` DDL; `blob_store.py`; `insights/postmortem.py` (honest-degradation pattern). Handoff triad = `37t.3` (voluntary reboot) + `gjg` (involuntary compaction) + resume briefs.

## 0. Tier decision (the load-bearing choice)

The snapshot is **irreplaceable raw capture** — the harness summarizes-and-discards, so the pre-compact context is *not derivable from index.db*. Therefore it lives in the **durable source tier**, mirroring `raw_sessions`/`raw_hook_events`. The forensic diff is **recomputable** → index tier. The durable loss *record* → user tier (`assertions`). Four artifacts, four tiers:

| Artifact | Tier | Durability | Rationale |
|---|---|---|---|
| `compaction_snapshots` row + blob | `source.db` (v→3) | durable, irreplaceable | harness discards it; not re-derivable |
| `session_events` boundary marker | `index.db` | rebuildable | already exists (`event_type='compaction'`), extended by 4ts.5 |
| `compaction_loss` forensic result | `index.db` | rebuildable | pure diff of snapshot × post-boundary tail |
| durable loss record | `user.db` `assertions` | durable, irreplaceable | the recursion anchor (§5) |

Source is a **durable tier** → its schema change is an **additive numbered migration** (`storage/sqlite/migrations/source/003_*.sql`) behind a backup manifest, per the durable-tier regime. `index.db` changes edit canonical DDL + rebuild plan (no migration chain).

---

## 1. `CompactionEvent` schema/DDL + hashing

### 1a. Source tier — `compaction_snapshots` (new, durable)
Add to `storage/sqlite/archive_tiers/source.py` (`SOURCE_SCHEMA_VERSION` 2→3) with migration `003_compaction_snapshots.sql`:

```sql
CREATE TABLE IF NOT EXISTS compaction_snapshots (
    snapshot_id                TEXT PRIMARY KEY,          -- natural-key hash (see hashing)
    origin                     TEXT NOT NULL CHECK ({check("origin", Origin)}),
    session_native_id          TEXT NOT NULL,             -- source-coord identity (parsed session_id not yet known at capture)
    boundary_ordinal           INTEGER NOT NULL CHECK(boundary_ordinal >= 0),  -- nth compaction in this session
    capture_tier               TEXT NOT NULL CHECK(capture_tier IN ('hook','jsonl-boundary')),
    trigger                    TEXT,                      -- 'auto'|'manual'|... from PreCompact payload; NULL for fallback
    context_blob_hash          BLOB NOT NULL CHECK(length(context_blob_hash) = 32),  -- content-addressed snapshot bytes
    context_bytes              INTEGER NOT NULL CHECK(context_bytes >= 0),
    boundary_native_message_id TEXT,                      -- last pre-boundary message (source coords)
    payload_json               TEXT,                      -- non-context hook metadata (reason, matcher, custom_instructions)
    observed_at_ms             INTEGER NOT NULL,
    UNIQUE(origin, session_native_id, boundary_ordinal)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_compaction_snapshots_session
ON compaction_snapshots(origin, session_native_id, boundary_ordinal);
```

Also extend `blob_refs.ref_type` CHECK: `IN ('raw_payload','attachment','sidecar','snapshot')` (same migration). The snapshot blob registers a `blob_refs` row (`ref_type='snapshot'`, `ref_id=snapshot_id`) so it participates in the existing two-invariant blob GC (lease + snapshot-reference check) and is never reclaimed while a snapshot row references it.

### 1b. Index tier — `compaction_loss` (new, rebuildable)
Add to `storage/sqlite/archive_tiers/index.py` (index `SCHEMA_VERSION` bump; batch with any other pending index bumps per fresh-first doctrine):

```sql
CREATE TABLE IF NOT EXISTS compaction_loss (
    loss_id           TEXT GENERATED ALWAYS AS (session_id || ':' || boundary_position) STORED UNIQUE,
    session_id        TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    boundary_position INTEGER NOT NULL CHECK(boundary_position >= 0),
    snapshot_id       TEXT,                       -- NULL => degraded (no_snapshot)
    capture_tier      TEXT,                       -- copied from snapshot for epidemiology stratification
    pre_item_count    INTEGER NOT NULL,
    retained_count    INTEGER NOT NULL,
    lost_count        INTEGER NOT NULL,
    transformed_count INTEGER NOT NULL,
    lost_items_json   TEXT NOT NULL DEFAULT '[]', -- ranked list: {item_class, key, ref, score, tier}
    measure_version   INTEGER NOT NULL,
    computed_at_ms    INTEGER NOT NULL,
    PRIMARY KEY(session_id, boundary_position)
) STRICT;
```
Boundary rows join to `session_events(event_type='compaction')` by `position`; the source→index link resolves at materialize time on `(origin, session_native_id, boundary_ordinal)`.

### 1c. Hashing (two distinct hashes; both SHA-256)
- **`context_blob_hash`** = SHA-256 of the snapshot payload bytes, written via `BlobStore.write_from_bytes` / `write_from_path`. **Free dedup**: repeated compactions of a growing session share large identical prefixes; identical content collapses to one blob (the store already no-ops on `dest.exists()`). Same addressing scheme as `raw_id`.
- **`snapshot_id`** = `sha256(f"{origin}\x00{session_native_id}\x00{boundary_ordinal}").hexdigest()` — a **natural-key content id**, matching Polylogue's "identity is computed, never stored redundantly" doctrine. Makes re-ingest idempotent and the id stable across rebuild. (Distinct from `context_blob_hash` so two structurally-identical contexts at different boundaries stay separately addressable.)

---

## 2. Loss-forensics diff algorithm (pseudocode)

Item universe = four classes, each with a normalizable **key** and an evidence tier (9l5.7):

| class | structural source | tier |
|---|---|---|
| `file-path` | `blocks.tool_path`, tool_input file args | structural |
| `tool-outcome` | `(tool_name, target, is_error, exit_code)` from the v16 keystone cols | structural |
| `marked-decision` | 37t.2 declared-marker notation in message text | structural (notation) / heuristic (prose) |
| `cited-ref` | regex over known ref grammars: `ObjectRef`/`EvidenceRef`, `bd:<id>`, `gh#N`, `file:line`, `session_id` | structural |

```
function compaction_loss_diff(session, boundary):
    # PRE = what existed before the harness discarded it
    snap = load_snapshot(session.origin, session.native_id, boundary.ordinal)
    if snap is None:
        # honest degradation, NOT fabrication (postmortem.py DegradedField pattern)
        return DegradedLoss(status="no_snapshot",
                            reason="no pre-compact snapshot captured for this boundary")
    pre_ctx = decode(blobstore.read_all(snap.context_blob_hash))   # hook: assembled context; fallback: reconstructed prefix

    # POST = what the model actually saw after the boundary (via 4ts.5)
    post_ctx = get_effective_context(session, at_position=boundary.position + 1)
               # == [materialized summary] + first-K post-boundary messages

    pre  = extract_items(pre_ctx)     # -> set[(class, key, attrs, ref, tier)]
    post = extract_items(post_ctx)

    pre_by_key  = index_by(pre,  k -> (k.class, normalize(k.key)))
    post_by_key = index_by(post, k -> (k.class, normalize(k.key)))

    retained, lost, transformed = [], [], []
    for key, item in pre_by_key:
        if key in post_by_key:
            if attrs_changed(item, post_by_key[key]):   # e.g. tool-outcome flipped, decision reworded
                transformed.append(item)
            else:
                retained.append(item)
        else:
            lost.append(item)

    # RANK lost by later-reference likelihood (the "will the agent need this again?" score)
    tail = session.messages_after(boundary.position)
    for item in lost:
        item.score = later_reference_likelihood(item, tail, corpus_base_rates)

    lost.sort(key=score, descending)
    return Loss(pre=len(pre), retained, lost, transformed,
                capture_tier=snap.capture_tier)   # tier stratifies epidemiology

function later_reference_likelihood(item, tail, base_rates):
    # signals, combined (weighted logistic; weights fit on labeled corpus, not hand-tuned magic):
    s_rederive = 1.0 if item.key reappears in tail else 0.0   # agent re-derived it => it was needed AND lost
    s_salience = 1.0 if item.class == 'marked-decision' else (0.6 if item.class=='cited-ref' else 0.3)
    s_baserate = base_rates[item.class]        # corpus prior: P(class referenced N turns after introduction)
    s_recency  = proximity_weight(item.position_in_pre)   # items near the boundary were most active
    return sigmoid(w1*s_rederive + w2*s_salience + w3*s_baserate + w4*s_recency)
```

`attrs_changed` is what separates **transformed** from **retained**: a `file-path` present in both but whose associated `tool-outcome` flipped error→ok, or a `marked-decision` whose text was compressed. This is the honest answer to "the summary *mentions* it but lost the *outcome*."

**Corpus epidemiology** (9l5.7 registered, `tier=structural`): aggregate `lost_count / pre_item_count` per class, per `capture_tier`, rendered as Wilson-interval proportions — e.g. *"compaction loses a median 34% (95% CI 29–39%) of marked-decisions; hook-tier snapshots and jsonl-boundary snapshots reported separately."* The measure registers with an explicit **construct caveat**: *"loss = verbatim structural absence in the post-boundary window, not semantic retention"* — so the number never overclaims that the model "forgot," only that the item is structurally absent.

---

## 3. Content-addressed snapshotting + honest degradation

**Capture path (source=hook, high fidelity):** `d1y`'s `hooks install` wires a `PreCompact` hook whose payload lands in `raw_hook_events` (durable). The ingest worker, on a `PreCompact` event: writes the payload's context bytes to the blob store (`write_from_bytes` → `context_blob_hash`, free dedup), inserts a `compaction_snapshots` row with `capture_tier='hook'`, `trigger` from the payload, and a `blob_refs` `ref_type='snapshot'` row.

> **Build-time verification gate (from bead notes + Risk 1):** the current Claude Code `PreCompact` payload MUST be inspected before implementation — if it carries only a *trigger/reason* and not the assembled context, `capture_tier='hook'` degrades to recording metadata only, and the pre-state is the JSONL fallback. The table's `capture_tier` column exists precisely to record this honestly rather than claim exact-context fidelity we cannot obtain.

**Fallback path (source=jsonl-boundary, honest degradation):** with no hook (or a metadata-only payload), the pre-state is the **composed prefix reconstructed from JSONL up to the boundary** — using existing lineage recomposition (`session_links` parent-up-to-branch + child tail) and `4ts.5`'s `boundary_start/end_position`. Serialize that prefix, blob-store it, insert the row with `capture_tier='jsonl-boundary'`, `trigger=NULL`. It is a real snapshot, explicitly labeled as *reconstructed transcript, not the exact assembled context* — the measure registry stratifies on `capture_tier` so the two are never silently pooled.

The snapshot ref uses the **already-existing** `ObjectRefKind` `"context-snapshot"` (`core/refs.py:27`) — no new ref kind needed.

---

## 4. `compaction_forgot(session_id)` MCP tool

Add to `mcp/server_*.py`; register in `EXPECTED_TOOL_NAMES` + `TOOL_CONTRACT` (`tests/infra/mcp.py` `ln`) or `test_tool_discovery`/`test_envelope_contracts` fail (known gotcha). Returns per compaction boundary:

```
{ session_id, boundaries: [ {
    boundary_position, ordinal, capture_tier,
    counts: { pre, retained, lost, transformed },
    lost_ranked: [ { item_class, summary, score, ref, tier } ],   # ref is resolve_ref-expandable
    construct: { measure_version, tier:"structural",
                 caveat:"structural absence, not semantic retention" }
} ] }
```
Honest-degradation branch mirrors `postmortem.py`: when `snapshot_id IS NULL`, emit `{ status:"no_snapshot", reason:"…" }` — never a fabricated zero. Every headline count carries ≥1 `EvidenceRef`, bounded by `_MAX_AGGREGATE_EVIDENCE`. A `read --view compaction_forgot` CLI surface auto-registers via `read_view_registry`.

---

## 5. Durable loss record + the compaction-proof recursion

The forensic result in `compaction_loss` is **rebuildable** (index tier) — it vanishes on `ops reset --index`. The *durable* record is written to **`user.db` `assertions`** with a new `AssertionKind` token (schema-free `TEXT`, e.g. `compaction_loss`; **regenerate `render openapi` + `render cli-output-schemas`** — the enum is embedded there). One assertion per high-score lost item: `scope_ref` = the session, `author_ref` = `insight:compaction-forensics`, body = the item's `resolve_ref` (refs-not-bodies, 37t.11 trust class **SYSTEM** — no verbatim prose stored), `context_policy_json` gates re-injection.

**Recursion closure — the loss record survives the *next* compaction because it is external to the transcript:**
1. Compaction N loses item X → forensics ranks it high → durable assertion written to `user.db`.
2. `user.db` is durable and *outside* the conversation transcript, so it survives (a) `ops reset --index` rebuilds and (b) the harness's summarize-and-discard at compaction N+1.
3. The **re-grounding ContextSource** (registers against `37t.11`, moment=`pre-compact-resume` / `SessionStart(source=compact)`) reads these assertions and re-injects the top-K lost-but-referenced items as refs (~200 token budget, keyed to measured loss → `jgp`-compliant). So X, lost at N, is restored before N+1; and its loss *record* is itself compaction-proof. The re-grounding arm is defined as an `ExperimentSpec` (`stc`): compact sessions with vs without re-grounding; the `37t.11` ledger rows are the arm evidence.

---

## 6. Bead breakdown (children of `gjg`; each with acceptance)

Sequencing: `gjg.1 → gjg.2 → gjg.3 → {gjg.4, gjg.5} → gjg.6`; `gjg.7` any time. All read-side work depends on `4ts.5` (`get_effective_context`, boundary ranges). *(I did not create these; proposed for the operator.)*

- **gjg.1 — Snapshot capture (source tier + blob + hook + fallback).** *AC:* migration `003` adds `compaction_snapshots` + `snapshot` `blob_refs` type behind a backup manifest; a live `PreCompact` on the operator machine lands a `capture_tier='hook'` row with a dedup-verified blob **OR** the `jsonl-boundary` fallback lands a labeled row for a known compaction session; second ingest of same boundary is idempotent (natural-key id). *Blocked by `d1y` for hook wiring; fallback independent.*
- **gjg.2 — Structural item extractor.** *AC:* `extract_items()` over the four classes returns tier-tagged keys from a fixture pre/post context; `file-path`/`tool-outcome`/`cited-ref` are `tier=structural` (read from `tool_path`/keystone cols/ref grammars, never regex-guessed from prose); property test: extraction is deterministic and origin-agnostic.
- **gjg.3 — Loss-forensics measure + epidemiology.** *AC:* `compaction_loss` populated by the §2 diff; retained/lost/transformed partition is exhaustive and disjoint (property test); registered in the 9l5.7 measure registry `tier=structural` with the semantic-retention caveat; a corpus run renders a per-class, per-`capture_tier` Wilson-interval epidemiology table.
- **gjg.4 — `compaction_forgot` MCP tool + CLI view.** *AC:* tool in `EXPECTED_TOOL_NAMES` + `TOOL_CONTRACT`; discovery + envelope tests pass; `no_snapshot` degradation path asserted; every count drillable via `resolve_ref`; `read --view compaction_forgot` auto-registers.
- **gjg.5 — Durable loss record + recursion.** *AC:* high-score lost items write `AssertionKind` rows (refs-only, trust class SYSTEM); `render openapi`/`render cli-output-schemas` regenerated; a test proves a record written pre-`ops reset --index` is present post-rebuild AND is retrievable after a simulated second compaction of the same logical session.
- **gjg.6 — Re-grounding ContextSource + experiment.** *AC:* a `ContextSource` (moment=`pre-compact-resume`) proposes top-K lost-but-referenced items under the flag only; deterministic assembly (property test); assembled output never exceeds moment budget; ledger rows written; the two-arm comparison is committed as an `ExperimentSpec` (`stc`). *Depends `37t.11`.*
- **gjg.7 — Handoff-triad doc.** *AC:* the `37t` epic description carries the OS-vision map: voluntary handoff (`37t.3`) / involuntary compaction (`gjg`) / cross-session resume (briefs), each with its owning bead.

---

## 7. Test strategy

- **Unit (pure, no I/O — `postmortem.py` model):** `extract_items` per class; `compaction_loss_diff` retained/lost/transformed on a hand-built pre/post fixture; `later_reference_likelihood` monotonic in each signal; `no_snapshot` degradation returns a `DegradedField`-style record, never zeros.
- **Property (Hypothesis):** partition exhaustive+disjoint (`retained ∪ lost ∪ transformed == pre`, pairwise empty); extraction deterministic + idempotent; ranking stable under permutation of input order; `snapshot_id` natural-key idempotency under re-ingest.
- **Storage:** migration `003` round-trip; `blob_refs ref_type='snapshot'` survives blob GC while referenced and is reclaimed when the snapshot row is deleted; `compaction_loss` FK-cascades on session delete; index rebuild regenerates `compaction_loss` byte-identically but leaves the `user.db` record intact.
- **Recursion test:** write durable record → `ops reset --index` → assert record present; simulate compaction N+1 → assert re-grounding source proposes it.
- **MCP:** discovery (`EXPECTED_TOOL_NAMES`), envelope contract, `resolve_ref` on an emitted `context-snapshot`/loss ref.
- **Clock hygiene:** all timestamp assertions via `frozen_clock`. **Demo path** (private-data-free): seed a synthetic 2-compaction session via `SessionBuilder` + `polylogue demo`.

---

## 8. Top-3 risks

1. **PreCompact payload may not carry the assembled context** (only a trigger/reason). Then hook-tier fidelity ≈ jsonl-boundary fidelity and the snapshot's marginal value over JSONL collapses. *Mitigation:* build-time payload verification (bead notes mandate it); `capture_tier` records the truth; never claim exact-context fidelity the payload can't provide.
2. **Construct validity of "loss":** structural absence ≠ the model forgot — the summary may semantically retain a reworded fact, inflating `lost_count` with false positives. *Mitigation:* the `transformed` bucket absorbs reworded/attribute-changed items; the measure registers `tier=structural` with an explicit "not semantic retention" caveat; the re-derivation signal (`s_rederive`) grounds "truly needed" in observed downstream behavior, not the extractor's opinion.
3. **Snapshot volume + privacy:** pre-compact context is the largest and most sensitive in-flight content; a durable irreplaceable table can grow unbounded, and a durable `user.db` record embedding verbatim prose would be an injection/exfil hole. *Mitigation:* blob dedup caps volume (shared prefixes collapse); durable records store **refs only** (37t.11 trust class SYSTEM); snapshot blobs join the existing two-invariant GC; add an operator-facing retention/purge policy for snapshot blobs (durable, but TTL-eligible on explicit consent).

**Key existing-surface reuse:** `ObjectRefKind "context-snapshot"` already exists (`refs.py:27`); `blob_store.py` gives content-addressing + dedup for free; `session_events(event_type='compaction')` is the boundary marker; `AssertionKind` is schema-free (`TEXT`) so the durable record needs no user-tier version bump; `4ts.5`'s `get_effective_context` supplies the post-boundary "what the model saw" side of the diff.

---

## [ad15b898d1eeef61f] SPEC — Cost-accounting correctness overhaul (bead `polylogue-f2qv`)

I have full grounding. Here is the spec.

---

# SPEC — Cost-accounting correctness overhaul (bead `polylogue-f2qv`)

Scope: the five named modules plus the two rollup sites that consume them. Grounded in live source; all file:line anchors verified. This is the cross-cutting correctness leg of epic `f2qv` (children `f2qv.1`–`f2qv.5`, plus siblings `5hf`, `ivsc`, `xy95`). It does **not** re-own the fork/resume inherited-prefix dedup (that is `4ts`); it consumes `4ts`'s logical grain.

## 0. Confirmed defect map (evidence)

| # | Defect | Anchor | Effect |
|---|---|---|---|
| D1 | Day/week summaries sum **physical** profiles | `session_summaries.py:98-104` (`for profile in profiles: total_cost += … total_wall += …`) | ~32% lineage double-count of cost, duration, messages, words |
| D2 | Cost-rollup insights sum `total_usd` over **physical** session insights | `archive_rollups.py:aggregate_cost_rollup_insights` (`for insight in insights: total_usd += estimate.total_usd`) | same lineage double-count in `cost_rollups`/`session_costs` |
| D3 | Exact provider-cost path is **dead** | `pricing.py:615-617` `_session_level_estimate` → `return None`; `session_reported_costs` table is written (`write.py:617`) but **read by nothing** (grep: zero readers) | `estimate_session_cost` never yields session-level `exact`; the whole `compute_session_cost` `status=="exact"` branch (`cost_compute.py:36-70`) is unreachable |
| D4 | Missing credit rates for current Opus/Sonnet | `subscription_pricing.py:84-105` has only `opus-4-6/4-5`, `sonnet-4-6/4-5`, `haiku-4-5`; `PRICING` curated set (`pricing.py:204-210`) has `opus-4-8/4-7`, `sonnet-…` | `get_credit_rate("claude-opus-4-8")→None` → `credit_cost=0` → subscription spend silently **$0** for every current-model session |
| D5 | `$/credit` hardcodes Pro tier | `cost_compute.py:133` `credit_cost / 21_700_000 * 20.0` | Max-5x/20x operators get a wrong subscription-USD figure; no tier parameter |
| D6 | Codex disjoint-lane decomposition has **no regression guard** | `write.py:2608 _provider_usage_disjoint_lanes` (fixed in 3938bc6c2, unguarded) | 7.69× inflation class can silently regress |
| D7 | `wall_duration_ms` summed across parallel subagents | `session_summaries.py:101` `total_wall += profile.wall_duration_ms` | >86.4M ms/day is representable (physically impossible); parallel subagent walls are additively double-counted |
| D8 | Enrichment downgrades EXACT→catalog | `cost_enrichment.py:54` guard only keeps stored estimate when re-derived is *not* confident; stored `exact` + re-derived `priced` (catalog) → **replaces exact with catalog** | provider-reported exact cost lost on read |

Note on the "5× output" item: current credit ratios (`output = 5×input`: Opus 10/50, Sonnet 6/30, Haiku 2/10) are **correct** per `reference_claude_subscription_credit_pricing` (output is 5× input, cache-read free on subscription). The live regression is **D4/D5**, not the ratio. The overhaul must **lock** the 5× ratio with a test so it can't regress, and fix the missing-rate/tier bugs that currently zero the subscription view.

---

## 1. Precedence lattice + schema touched

### 1a. Cost-basis precedence (per model-attributed unit; never downgrade EXACT)

```
provider_reported (status=exact)   ≻   catalog_priced (status=priced)   ≻   heuristic_estimated (status=partial/estimated)   ≻   unavailable
```

Rules:
- **Monotone floor:** a unit already at `provider_reported` may only be re-priced *into another provider_reported* (newer catalog stamp), never demoted to `catalog_priced`/`heuristic`. Enrichment and rebuild both obey this.
- The three cost *views* are **orthogonal outputs computed at every tier**, not competing sources:
  - `api_equivalent_usd` — list-price value (cache reads at list). From `provider_reported_usd` when exact, else `catalog_priced_usd`.
  - `subscription_credit` (integer credits) + `subscription_usd` — credit formula with cache-**reads zeroed** on subscription tiers, `$/credit` from the operator tier.
  - Legacy `total_usd` = `api_equivalent_usd` (compat).
- The lattice is realized by a single `resolve_basis(unit)` that stamps `cost_provenance ∈ {provider_reported, catalog_priced, heuristic_estimated, unavailable}` and `price_snapshot_version`.

### 1b. Rollup grain: `LOGICAL_SESSION`

All day/week/cost rollups aggregate over **logical sessions** (`profile.logical_session_id or profile.session_id`), taking **one representative** per logical id — the high-water/canonical physical session per `4ts` — never a sum across physical children. This is the measure-algebra convergent design: cost, credits, tokens, messages, words are `SUM over distinct logical`, wall is interval-union (§2e), `session_count` stays physical, `logical_session_count` stays the distinct count.

### 1c. Schema touched (durability-classified)

| Tier | Object | Change | Regime |
|---|---|---|---|
| `index.db` (24→25, **rebuildable**) | `session_profiles` | ensure `cost_credits`, `cost_subscription_usd`, `cost_provenance`, `price_snapshot_version` populated via the corrected path (columns exist; semantics change) | derived → DDL edit + rebuild plan, no migration helper |
| `index.db` | `session_model_usage` | per-model rows keyed at **event grain** (partition, not repeated session total — `f2qv.1`/#2472); carry `materializer_version` (`f2qv.5`) | derived → rebuild |
| `index.db` | `session_reported_costs` | now **read** by `_session_level_estimate` (wire the dead path) | derived |
| `user.db` (**durable**) | operator subscription tier (`pro`/`max_5x`/`max_20x`) | store as a `user_settings`/`assertion` metadata row (`AssertionKind.METADATA`, key `subscription_tier`) — plain `TEXT`, **no numbered migration** needed since the assertions table is schema-free. If (and only if) it must be a typed `user_settings` column, add additive `migrations/user/NNN_subscription_tier.sql` behind a backup manifest. | durable → prefer assertion; migration only if columnar |

Because index.db is rebuildable, the grain/basis/lane corrections ship as a **canonical-DDL + materializer_version bump + rebuild** (`polylogue ops reset --index && polylogued run`), per CLAUDE.md schema regimes — **no upgrade helper** (`devtools lab policy schema-versioning` would reject one).

---

## 2. Fix algorithms (pseudocode)

### 2a. Wire the dead exact path (D3) — `pricing.py:_session_level_estimate`

```
def _session_level_estimate(session):
    row = read_session_reported_costs(session.id)        # currently unread table
    if row is None or row.total_usd is None:
        return None
    usage = CostUsagePayload(row.input, row.output, row.cache_read, row.cache_write, row.total)
    return _exact_estimate(source_name=session.origin, total_usd=row.total_usd,
                           session_id=session.id, model_name=row.model_name, usage=usage)
    # _exact_estimate already fills basis.provider_reported_usd + parallel catalog_priced_usd
```

`estimate_session_cost` already prefers a session-level `exact`/`priced` (`pricing.py:635, 716`) once this returns non-None → the `compute_session_cost` exact branch becomes reachable.

### 2b. Per-model credit rates (D4) — `subscription_pricing.py:MODEL_CREDIT_RATES`

Add every curated Anthropic key from `_CURATED_PRICING` with the 5×-input ratio, derived, not hand-typed drift:

```
# Derive credit rate from the same $/1M schedule the API view uses:
# input_credits = round(input_usd_per_1m / ANCHOR)      # ANCHOR = Haiku-input $0.8 → 2 credits ⇒ ANCHOR=0.4
# output_credits = 5 * input_credits   (locked ratio)
opus-4-8/4-7/4-6/4-5: (10, 50)   # $15 in → 10, 5× → 50
sonnet-4-8/4-7/4-6/4-5: (6, 30)
haiku-4-5: (2, 10)
cache_read_credits = 0            # free on subscription (invariant)
cache_write_credits = input_credits   # cache write bills at input rate
```
Property test asserts `MODEL_CREDIT_RATES.keys() ⊇ {curated Anthropic keys in _CURATED_PRICING}` and `output_credits == 5 * input_credits` for every row.

### 2c. Tier-parametrized `$/credit` (D5) — `cost_compute.py:133` + `subscription_pricing.py`

```
def subscription_usd(credits, tier: SubscriptionTier):
    return round(credits / tier.credit_pool * tier.monthly_fee_usd, 6)

tier = load_operator_tier()          # user.db assertion 'subscription_tier'; default SUBSCRIPTION_TIERS['pro']
sub_equivalent = subscription_usd(credit_cost, tier)   # replaces / 21_700_000 * 20.0
```
`credit_pool`/`monthly_fee_usd` already parametrized in `SUBSCRIPTION_TIERS` (`subscription_pricing.py:27-49`).

### 2d. Enrichment never downgrades EXACT (D8) — `cost_enrichment.py:54`

```
STRENGTH = {"exact":3, "priced":2, "partial":1, "estimated":1, "unavailable":0}
if STRENGTH[insight.estimate.status] > STRENGTH[rederived.status]:
    return insight            # keep the stronger stored basis
if insight.estimate.cost_provenance == "provider_reported" and rederived.cost_provenance != "provider_reported":
    return insight            # never trade provider_reported for catalog
return graft(rederived)       # only augments upward or laterally
```

### 2e. Logical-grain rollup + wall interval-union (D1, D2, D7) — `session_summaries.py`, `archive_rollups.py`

```
def summarize_day(profiles, target_date):
    by_logical = {}
    for p in profiles:
        lid = p.logical_session_id or p.session_id
        by_logical.setdefault(lid, []).append(p)

    reps = [logical_representative(group) for group in by_logical.values()]   # 4ts canonical/high-water
    total_cost   = sum(r.total_cost_usd  for r in reps)     # was: over all physical
    total_credit = sum(r.cost_credits    for r in reps)
    total_msgs   = sum(r.message_count   for r in reps)
    total_words  = sum(r.word_count      for r in reps)

    # wall: union of [start,end) intervals across ALL physical members
    # (parallel subagents overlap in real time; summing double-counts, D7)
    intervals = [(p.first_message_at, p.last_message_at) for p in profiles if p.spans_wall]
    total_wall_ms = interval_union_ms(intervals)           # merge-overlaps then sum lengths
    # invariant: total_wall_ms <= 86_400_000 for a single calendar day

    return DaySessionSummary(session_count=len(profiles),
                             logical_session_count=len(by_logical),
                             total_cost_usd=total_cost, total_credit=total_credit,
                             total_wall_duration_ms=total_wall_ms, ...)
```
`interval_union_ms`: sort by start, sweep-merge overlapping `[start,end)`, sum merged lengths. Same logical-dedup applied in `aggregate_cost_rollup_insights` (group insights by logical id first, take representative, then group by `(source, model)`).

### 2f. Codex disjoint-lane assertion at parse (D6) — `write.py:_provider_usage_disjoint_lanes`

Add an in-writer invariant (raise/log-and-clamp) enforcing the four disjoint lanes so a regression is caught at materialize, not in aggregate cost:
```
assert cached_input <= input_total                     # cached ⊆ input (Codex 100%)
assert reasoning_output <= output_total                # reasoning ⊆ output (Codex 98.9%)
fresh_input      = max(input_total - cached_input, 0)
cache_read       = cached_input
output_completion= max(output_total - reasoning_output, 0)
output_reasoning = reasoning_output
# store four DISTINCT lanes; never fold cache_read into fresh_input
# guard: fresh_input + cache_read == input_total  AND
#        output_completion + output_reasoning == output_total
```

### 2g. Single pricing source / last-segment resolve (`f2qv.4`) — `pricing.py:_normalize_model`

Current resolver uses `removeprefix` + longest-`startswith` (`pricing.py:325-347`) — tighten to strict **last-path-segment** match (`model.split("/")[-1]`, then date-suffix strip, then exact-key lookup, unknown→labelled `no_price`, never silent second table). Confirm no `tokencost` import survives (grep gate). Curated overrides remain the only second map and are hand-verified, not drift.

---

## 3. Migration

1. **index.db (derived, rebuildable):** bump canonical index schema `24→25`; bump `SESSION_INSIGHT_MATERIALIZER_VERSION` and add a `provider_usage` materializer version (`f2qv.5`). No migration chain — edit canonical DDL + `docs/plans` rebuild plan. Deploy = `polylogue ops reset --index && polylogued run`. The `f2qv.5` self-heal gate makes existing `session_model_usage` rows re-derive on a daemon drain without a manual `maintenance rebuild-index`.
2. **user.db (durable):** operator subscription tier as an `AssertionKind.METADATA` row `subscription_tier=<pro|max_5x|max_20x>` (schema-free `TEXT` — no numbered migration). Provide a CLI/MCP setter; default `pro`. Only if a typed column is mandated: additive `migrations/user/NNN_subscription_tier.sql`, one `PRAGMA user_version` step, behind a verified backup manifest.
3. **No durable destructive change.** `session_reported_costs` is already populated historically; wiring the reader is read-only. If historical rows are absent for old sessions, `_session_level_estimate` returns None → clean fallback to catalog (no crash).
4. Regenerate generated surfaces: `render openapi`, `render cli-output-schemas` (new `subscription_usd`/`cost_provenance` fields), `render topology-projection` if any module added, then `render all --check` (grep for `out of sync`).

---

## 4. Test / invariant suite

Property/invariant tests (Hypothesis where shaped), each citing its bead:

1. **Lineage non-double-count (D1/D2, `4ts`+`f2qv`):** synthetic day with one logical session physically replayed 3× → `summarize_day.total_cost_usd == single-logical cost` (not 3×); `logical_session_count==1`, `session_count==3`.
2. **Per-model partition (#2472, `f2qv.1`):** synthetic two-model session → `sum(per_model.tokens) == session_total` and no model row carries the full session total.
3. **Disjoint lanes / 7.69× guard (D6, `f2qv.2`):** synthetic Codex `token_count` with cached⊆input, reasoning⊆output → four lanes disjoint, sum to totals; assert `raw(input+output) > billable(fresh_input+output)` (the inflation repro stays green). Same for Claude `cache_read/cache_creation`.
4. **Credit-rate coverage (D4):** `MODEL_CREDIT_RATES.keys() ⊇ curated Anthropic keys`; `get_credit_rate("claude-opus-4-8")` is non-None; every rate `output==5×input`, `cache_read==0`.
5. **Tier parametrization (D5):** same credits under `pro` vs `max_20x` yield different `subscription_usd`; `pro` reproduces `credits/21.7M*20`.
6. **Dual view divergence (`f2qv.3`):** cache-heavy session → `subscription_usd < api_equivalent_usd` (cache reads free); both fields present and distinct.
7. **No EXACT downgrade (D8):** stored `exact` provider_reported + re-derived `priced` catalog → `enrich_*` keeps the exact estimate.
8. **Exact path live (D3):** a session with a `session_reported_costs` row → `estimate_session_cost.status=="exact"` and `compute_session_cost` takes the exact branch.
9. **Wall interval-union (D7):** two overlapping parallel subagents (00:00–00:10, 00:05–00:15) → `total_wall==15min` not 20; day-level `total_wall_ms <= 86_400_000`.
10. **Single pricing source (`f2qv.4`):** grep gate `tokencost` absent from deps+imports; every live-archive model resolves via last-segment or is labelled `no_price` (no silent second table).
11. **Contract-backing (`f2qv` AC7):** the `docs/internals.md` "cache lanes never folded" prose asserted by test 3.

Verification cadence: `devtools test <changed files>` (testmon-affected), then `devtools verify` once pre-PR; live-archive spot check `polylogue analyze usage --origin codex-session` + `cost_rollups` MCP for dual views without 7.69× inflation (respect `xy95` — gate the expensive full-detail path).

---

## 5. Bead breakdown (maps onto existing `f2qv.*`; 6 units)

| Bead | Title | Acceptance |
|---|---|---|
| **B1** `f2qv.2` (adopt) | Disjoint-lane normalizer + regression guard | Four labelled lanes sum to totals; disjointness invariant test; 7.69× repro green; internals.md contract test-backed |
| **B2** `f2qv.1` (adopt) | Per-model rollup partition (#2472) | Two-model session partitions exactly; sum(per-model)==session total; regression test cites #2472 |
| **B3** `f2qv.4` (adopt) | Single LiteLLM pricing source, last-segment resolve, drop tokencost | grep shows no tokencost/second map; every live model resolves-or-labelled-unknown; resolver test |
| **B4** `f2qv.3` (adopt) + D3/D4/D5 | Dual cost view + wire exact path + credit rates + tier `$/credit` | `_session_level_estimate` reads `session_reported_costs`; `MODEL_CREDIT_RATES ⊇ curated keys`; `subscription_usd(tier)` replaces Pro hardcode; `subscription_usd < api_equivalent_usd` on cache-heavy session; 5× ratio locked; no EXACT downgrade in enrichment |
| **B5** (new child) `f2qv.6` | Logical-grain rollups + wall interval-union | Day/week/cost-rollup dedup to logical grain (test 1); `total_wall_ms<=86.4M/day`, interval-union (test 9); `session_count` physical, `logical_session_count` distinct |
| **B6** `f2qv.5` (adopt) | Version-gate provider-usage projection self-heal | `session_model_usage` carries materializer_version; version bump auto-refreshes on daemon drain (no manual rebuild); archive_debt zero-token rows drain; paged, no D-state |

Sequencing: B1→B2 (lanes correct before partition), B3 (pricing) parallel; B4 depends on B1+B3; B5 depends on `4ts` logical grain + B4 (dedup correct cost); B6 last (self-heal the corrected materializer). Sibling `xy95` (perf) and `ivsc` (state_5 drift classify) stay separate.

---

## 6. Top-3 risks

1. **Logical-representative selection is the whole correctness hinge and is owned by `4ts`, not here.** "One representative per logical id" requires `4ts`'s high-water/canonical physical-session rule to be live and correct; if B5 lands before `4ts`'s `logical_representative` is trustworthy, the dedup either under-counts (drops a divergent tail's real cost) or re-introduces double-count. Mitigation: gate B5 on `4ts` logical grain; test 1 must use a real fork/resume fixture, not a naïve duplicate.

2. **Grain change is a live rebuild of the 38 GB index, not a code-only fix.** Basis/lane/grain corrections change materialized `session_profiles`/`session_model_usage` semantics → require `ops reset --index && polylogued run` on the active archive; a partial/interrupted rebuild leaves mixed old-basis and new-basis rows read as drift (`xy95` stale-rollup class). Mitigation: bump `materializer_version` so the `f2qv.5` self-heal drain re-derives rather than leaving stale rows; batch same-tier bumps (CLAUDE.md) — don't reset per sub-bead.

3. **Subscription tier is operator state, and a wrong/absent tier silently mis-scales every subscription-USD figure.** Defaulting to `pro` when the operator is on Max-20x understates `subscription_usd` ~18×; storing it as a schema-free assertion means a typo (`max20x` vs `max_20x`) reads as unset. Mitigation: closed-vocabulary validation on the setter, explicit "tier unconfigured" caveat (reuse `subscription_unconfigured` reason in `pricing.py:CostUnavailableReason`) rather than a silent Pro default, and surface the assumed tier in every subscription-view payload.

---

## [a175db16d46c3ccd8] Spec: Config + Runtime-Preferences Engine (w8db / spine y4c)

I have complete grounding. Here is the spec.

---

# Spec: Config + Runtime-Preferences Engine (w8db / spine y4c)

## 0. Grounding (what exists today)

- **`polylogue/config.py`** already implements a **5-layer deployment resolver**: `_default_config_values()` → site TOML → user TOML → `POLYLOGUE_*` env → CLI overrides, via `load_polylogue_config()`. `PolylogueConfig` carries `.layers`/`.layer_of(key)` provenance. `_CONFIG_INVENTORY` (≈60 `ConfigInventoryEntry` rows) is the *deployment* key registry with `owner_class`/`reload_behavior`/`toml_path`/`env_var`. Secret redaction (`is_secret_config_key`, `SECRET_CONFIG_KEYS`) is mature. **This is the deployment half — it does not touch user.db and has no `db` layer.**
- **`user_settings` table exists but is DEAD**: declared in `user.py` (`USER_SCHEMA_VERSION=4`) + migration `004_user_settings.sql`, but grep finds **no read/write helper anywhere**. It is unwired and empty in practice — this is the seam w8db unifies away.
- **`assertions` table** (user.py) already has every column a scoped setting needs: `scope_ref`, `target_ref`, `key`, `kind` (TEXT, no CHECK), `value_json`, `author_ref`/`author_kind`, `context_policy_json` (`{"inject":false}` default), `staleness_json`, `status`, `confidence`, `supersedes_json`. Write path is `upsert_assertion(conn, *, assertion_id, target_ref, kind, scope_ref, key, value, author_ref, author_kind, context_policy, staleness, …)` with deterministic ids. `AssertionKind` (core/enums.py:399) is a closed enum, SQL is TEXT so new members need **no DDL** — but the enum is embedded in `render openapi` + `render cli-output-schemas` (regen required).
- **Migration runner** (`storage/sqlite/migration_runner.py`): durable tiers (SOURCE, USER) use numbered `NNN_name.sql` + a required **manifest** (backup gate); next user migration is **005**.
- **DSL** (`archive/query/expression.py`): Lark grammar, macro/`@name` expansion must happen **before lowering** (fnm.12 requirement), predicate-groups composable in `find`/unit-where/pipeline.

**Layering constraint:** `config.py` is pure filesystem/env and cannot reach user.db. The runtime-pref resolver is a **new product-layer module** (proposed `polylogue/runtime_prefs/`) that composes `PolylogueConfig` (deployment) with a user.db read through `SessionRepository`/`services` — surfaces must not import substrate internals directly (`layering.yaml`).

---

## 1. Schema / DDL + tier

**Tier: `user.db`** (durable, irreplaceable). `USER_SCHEMA_VERSION 4 → 5`.

**Two new `AssertionKind` members** (enum-only, TEXT column, no CHECK — schema-free per doctrine):
- `SETTING = "setting"` — a runtime preference row.
- `SETTING_SUGGESTION = "setting_suggestion"` — a learned-default candidate (1jc). *(Evaluate reusing `TRANSFORM_CANDIDATE`/setup_improvement machinery from 37t.10 before adding — the "every kind needs a surface" cost is real. Recommend a distinct kind for clean judge filtering.)*

**Setting row encoding** (reuses `assertions` columns 1:1):

| Column | Setting semantics |
|---|---|
| `kind` | `"setting"` |
| `key` | registry pref key, e.g. `read.default_view`, `query.default_window`, `macro:mine`, `preset:skeleton` |
| `scope_ref` | ObjectRef: `scope:global` \| `scope:repo:<name>` \| `scope:origin:<token>` \| `scope:surface:<cli\|mcp\|web\|hook>` |
| `target_ref` | `scope_ref` (setting targets its own scope node) |
| `value_json` | typed value validated against the registry entry |
| `author_ref` / `author_kind` | **actor chain**: `author_kind ∈ {operator, agent, harness}` (operator ≈ existing `user`) |
| `context_policy_json` | `{"inject":false, "pin":bool, "important":bool}` — pin/`!important` semantics |
| `staleness_json` | learned-default decay: `{"origin":"learned","evidence_ref":…,"half_life_days":N,"reinforced_at_ms":…}` |
| `status` | `active` for effective rows; `candidate/accepted/rejected` for suggestions |
| `confidence` | learned-default dominance score (1jc) |

**Deterministic id:** `assertion-setting:sha256(scope_ref \0 key \0 author_kind)` — one row per (scope, key, actor), upsert-idempotent.

**Migration `005_settings_unify.sql`** (+ manifest, backup-gated):
```sql
-- Copy-forward any legacy user_settings rows into assertions as global settings.
INSERT OR IGNORE INTO assertions
  (assertion_id, scope_ref, target_ref, key, kind, value_json,
   author_ref, author_kind, status, visibility, context_policy_json,
   created_at_ms, updated_at_ms)
SELECT
  'assertion-setting:' || <sha256(scope:global,setting_key,operator)>,  -- computed in Python pre-pass
  'scope:global', 'scope:global', setting_key, 'setting', value_json,
  author_ref, 'operator', 'active', 'private', '{"inject":false}',
  updated_at_ms, updated_at_ms
FROM user_settings;

DROP TABLE user_settings;

-- Hot-path partial index for scope resolution.
CREATE INDEX IF NOT EXISTS idx_assertions_setting
  ON assertions(key, scope_ref) WHERE kind = 'setting';
```
Also drop the `user_settings` block from `USER_DDL` and `004_user_settings.sql` stays historical. **Drop is justified** — the table is provably unwired (no helper) so copy-forward is defensive, not migratory; this satisfies "copy-forward design + explicit consent" for a destructive durable change. (SHA id must be computed in a Python pre-INSERT pass, not raw SQL — hashing is not available in SQLite; do the copy-forward as a Python migration step keyed off the numbered SQL, matching the runner's step model.)

---

## 2. Resolution algorithms (pseudocode)

### 2a. Outer source precedence (fixed total order)
```
resolve(key, ctx) -> ResolvedValue:
    reg = REGISTRY[key]                       # typed entry; raises on unknown key
    candidates = []
    # highest → lowest, first non-missing wins UNLESS a pinned db row escalates
    if key in ctx.cli_flags:      return ResolvedValue(ctx.cli_flags[key], layer="flag")
    if reg.env_var in environ:    return ResolvedValue(coerce(environ[reg.env_var], reg.type), layer="env")
    if reg.is_deployment and key in toml_cfg: return ResolvedValue(toml_cfg[key], layer="file")
    db = resolve_db(key, ctx)                 # scope×actor resolution (2b)
    if db is not None:
        if db.pinned:  return ResolvedValue(db.value, layer="db", scope=db.scope, pinned=True)
        return ResolvedValue(db.value, layer="db", scope=db.scope, actor=db.actor)
    return ResolvedValue(reg.default, layer="default")
```
Precedence per y4c AC: **flag > env > file(toml) > db > default**. `pin` does *not* re-order across the source axis (a flag still wins over a pinned db row — the operator's live intent overrides their stored intent); `pin` operates **within** the db layer (2b) and against **learned defaults** (2c).

### 2b. Inner scope×actor resolution (within db layer)
```
resolve_db(key, ctx) -> DbSetting | None:
    rows = SELECT * FROM assertions
           WHERE kind='setting' AND key=:key AND status='active'
             AND scope_ref IN (scope_refs_for(ctx))          # single batched query
    matching = [r for r in rows if scope_matches(r.scope_ref, ctx)]
    if not matching: return None
    # Sort keys, descending: important > specificity > actor > recency
    return max(matching, key=lambda r: (
        r.important,                          # !important escalates over specificity
        SCOPE_RANK[scope_type(r.scope_ref)],  # surface(4)>origin(3)>repo(2)>global(1)
        ACTOR_RANK[r.author_kind],            # operator(3)>agent(2)>harness/learned(1)
        r.updated_at_ms,
    ))

scope_refs_for(ctx) = ["scope:global",
                       f"scope:repo:{ctx.repo}",
                       f"scope:origin:{ctx.origin}",
                       f"scope:surface:{ctx.surface}"]   # None-safe
```
Scope is a **defined precedence order among four node types** (per y4c design: global→repo→origin→surface), not an orthogonal matrix — v1 rows scope to exactly one node; composite scopes (`repo+surface`) deferred to v2 with evidence.

### 2c. Learned-default admission + recursive safety (1jc)
```
detect_learned_defaults():                    # low-freq insight pass, trailing 30d
    for (verb, flags, scope) in aggregate(invocation_spans WHERE not self_originated):
        if support(n) < REGISTRY['learn.min_support'](=20): continue        # threshold is itself a pref
        if dominance < REGISTRY['learn.min_dominance'](=0.70): continue
        key = map_flag_to_pref(verb, flags)
        if REGISTRY[key].is_deployment: continue                            # AC: deployment keys never proposed
        if has_pinned_row(key, scope): continue                            # user override protection
        if suppressed(key, scope): continue                                # rejected → no re-propose
        emit SETTING_SUGGESTION candidate {key, scope, value, evidence_agg, expected_effect}

# RECURSIVE-SAFETY SELF-EXCLUSION (the critical filter):
self_originated(span) := span.verb in {"config","judge"} or span.origin=="learned-default-apply"
```
- **Never auto-applies** — a suggestion is a candidate assertion judged in `polylogue judge`; accept writes the settings row **attributed to the evidence** (`staleness_json.evidence_ref`), so `config explain` answers "why is this my default?".
- **Staleness decay:** learned rows carry `half_life_days`; at resolve time a decayed learned row loses the actor tiebreak and, past a floor, is treated as absent (reverts to code default) unless reinforced. Operator/`important` rows never decay.

### 2d. `config explain <key>` — findings-as-object
Returns a `ConfigResolution` object (not a string): `{key, effective_value, winning_layer, winning_scope, winning_actor, pinned, learned{evidence, confidence, staleness}, shadowed:[{layer/scope/actor, value, reason_lost}]}`. The **shadowed chain** makes every losing candidate legible — this is the construct-honesty surface.

### 2e. Presets / `@name` namespace
- **Query macros** (fnm.12): stored as `kind=setting, key=macro:<name>`, value = DSL predicate-group text. Validated at *definition* time by compiling against the grammar (caret error on parse failure). `@name` in any predicate-group position expands **before lowering**, carrying macro-provenance spans for `explain`. Macros may reference macros: **cycle-checked + depth-capped at definition time**. v1 = predicate-groups only (no pipeline stages inside macros).
- **View/render/column/budget presets** (y8w/1lm): `key=preset:<name>`, value = JSON bundle `{view, fold_budget, columns, …}`. `@name` / `--preset name` resolves the bundle as a synthetic override layer **just below flags**. One `@name` namespace, dispatched by resolved kind (query-fragment vs pref-bundle).

---

## 3. Migration

- **Numbered durable migration** `005_settings_unify.sql` + manifest (runner enforces manifest for USER tier). Python copy-forward step computes deterministic ids (SHA unavailable in SQLite). Bump `USER_SCHEMA_VERSION → 5`.
- **Enum additions** `SETTING`, `SETTING_SUGGESTION` → **regenerate** `render openapi` + `render cli-output-schemas` + `render configuration.md` (new) + Nix module options (generated from registry). Watch the `render all --check` gotcha: grep for `out of sync`, it exits 1 while printing per-surface `sync OK`.
- **New module** `polylogue/runtime_prefs/` → run `devtools render topology-projection && render topology-status`, commit `topology-target.yaml` + `topology-status.md` or `render all --check` fails.
- No index.db/embeddings.db change (derived tiers untouched).

---

## 4. Test strategy

- **Resolution laws (property / Hypothesis):** source precedence is a total order (flag>env>file>db>default) for any candidate set; scope specificity monotonic (surface beats origin beats repo beats global); actor tiebreak deterministic; `pin` blocks learned override but not flag override; `!important` escalates within db layer only; `resolve()` is a pure function of (registry, ctx, rows).
- **Migration round-trip:** seed `user_settings` rows pre-005, migrate, assert exact copy-forward into `assertions(kind=setting, scope:global, author_kind=operator)`; `user_settings` gone; `USER_SCHEMA_VERSION=5` gate; manifest present (runner rejects otherwise). Protected: uses `frozen_clock`.
- **Macros/presets:** define→grammar-validate (caret on bad macro)→`@name` composes in find/unit-where/pipeline on live/demo archive; cycle + depth guards; `explain` shows expansion with provenance; completions surface `@`-names.
- **Learned-default recursive-safety:** seeded telemetry where the dominant pattern *is* `config`/`judge` invocations → assert **no** suggestion emitted (self-exclusion); dominant real pattern → correct candidate with evidence aggregate; accept-in-judge writes scoped row + live effect; reject suppresses re-proposal; staleness decay reverts a stale learned default; deployment/secret keys never proposed; suggestion cap enforced.
- **Registry validation lane** (new `devtools lab policy config-registry`): every key has default + owner surface + ≥2 legitimate values + scope-applicability set + type; **deployment-key set ∩ runtime-pref-key set = ∅**; secret keys are deployment-only (never db-tier); learned-eligible ⊆ runtime, never deployment.
- **`config explain` snapshot** (syrupy): the `ConfigResolution` object incl. shadowed chain, under two settings values.
- **Security:** secret keys rejected as `kind=setting` writes; MCP/web settings panel cannot read/write secrets; redaction preserved on all config surfaces.

---

## 5. Bead breakdown (spine decomposition; do not edit — propose)

Existing dependents (`3xx`, `y8w`, `6kh`, `1jc`, `fnm.12`) already carry execution-grade AC. The **engine work (y4c spine)** decomposes into three implementable sub-beads; the bundles then consume it.

1. **y4c-a — Settings substrate + migration.** `AssertionKind.SETTING`/`SETTING_SUGGESTION`; `upsert_setting`/`read_setting`/`list_settings_by_scope` helpers over `assertions`; migration 005 (copy-forward + drop `user_settings` + partial index); `USER_SCHEMA_VERSION 5`; regen openapi/cli-output-schemas/topology.
   *AC:* settings row round-trips through `upsert_assertion`; migration copies then drops `user_settings`; partial index used by `resolve_db`; render gates green.
2. **y4c-b — Scope×actor×override resolver + typed registry + surfaces.** `RuntimePrefRegistry` (declare-once typed keys with scope-applicability + learn-eligibility); `resolve()`/`resolve_db()`/`ConfigResolution`; `config get/set/effective/explain` (CLI + MCP + web panel); live effect via daemon event bus.
   *AC:* db-backed prefs work end-to-end for `read.default_view` + `query.injection_budget` with live effect; `config effective` names source per key; `explain` returns the shadowed chain; pin/!important laws hold.
3. **y4c-c — Doctrine + audit + Nix parity + validation lane.** Doctrine page in internals (a knob needs ≥2 values, a 90%-right default, an owner); audit existing `_CONFIG_INVENTORY` against it with a **deletion list**; deployment/runtime partition; Nix module options generated from registry + render-checked; `configuration.md` generated; `devtools lab policy config-registry` lane.
   *AC:* doctrine committed; deletion list merged; registry validation lane green; Nix options render-checked.
4. **fnm.12 — Presets/macros `@name`.** (existing) Definition-time grammar validation; expand-before-lower with provenance; cycle/depth guards; CLI+MCP define/list/delete; completions. *AC as in bead.*
5. **y8w / 6kh — Reading + query-scope bundles.** (existing) Consume y4c-b resolver: per-scope default view, fold budgets, columns, auto-read (y8w); implicit-scope macro layer + footer disclosure + `all:` widen (6kh). *AC as in beads.*
6. **1jc — Learned defaults.** (existing, gated on 37t judgment gate) Detector with min-support/dominance (themselves prefs), recursive-safety self-exclusion, `SETTING_SUGGESTION` candidates with evidence, judge integration, staleness decay, caps, deployment-key exclusion. *AC as in bead.*
7. **3xx — Verb/ops prefs.** (existing) Confirmations, judge defaults, copy formats, spend budget, quiesce — via daemon event bus for live effect. *AC as in bead.*

---

## 6. Top-3 risks

1. **Deployment/runtime key-space bleed + secret leakage.** A key present in both `_CONFIG_INVENTORY` (toml/env, redaction-aware) and the db-settings registry produces ambiguous precedence, and the db tier is *more visible* (MCP tools, web settings panel) than redacted config output — a secret promoted to a `kind=setting` row would bypass `SECRET_CONFIG_KEYS` redaction. **Mitigation:** hard registry partition invariant (deployment ∩ runtime = ∅), secret keys deployment-only, enforced by the y4c-c validation lane and a write-time guard in `upsert_setting`.
2. **Learned-default feedback loop / construct dishonesty.** Telemetry that includes `config`/`judge` invocations self-reinforces ("you configure a lot → here's a config default"); silently-changing learned behavior erodes trust in what the archive *observed* vs *decided*. **Mitigation:** self-exclusion filter (`self_originated`), pin protection, never-auto-apply (judge gate), evidence attribution so `explain` answers "why is this my default?", and staleness decay so stale inferences revert rather than calcify.
3. **Durable-tier destructive migration + hot-path cost.** Dropping `user_settings` is a destructive durable change — copy-forward correctness + manifest gate are load-bearing even though the table is empty in practice; and the `SETTING` enum addition silently breaks `render openapi`/`cli-output-schemas` (`render all --check` exits 1 while printing `sync OK`). Separately, every CLI invocation resolves N prefs — a per-key query would be a read-amplification regression. **Mitigation:** Python copy-forward step + manifest; regen-and-grep-`out of sync` in the migration PR; **single batched `resolve_db` query** over the `idx_assertions_setting` partial index loading all settings for the ctx scope-set at once, not per-key.

---

## [a45d20b65f3d9eb91] Wave-5 Spec: Search Relevance, Ranking Quality & Explainability

I have comprehensive grounding. Here is the implementation spec.

---

# Wave-5 Spec: Search Relevance, Ranking Quality & Explainability

## Grounding (verified against source)

| Concern | Real anchor | Current state |
|---|---|---|
| RRF fusion | `storage/search_providers/hybrid.py:47` `reciprocal_rank_fusion(*lists, k=60)` | equal-weight, positional rank, `1/(k+rank)`, stable tie-break on `item_id` |
| Hybrid scored search | `hybrid.py:137` `search_scored` | fetches `limit*3` per lane, FTS score = `1/(i+1)`, vector via `vector_provider.query` |
| Score decomposition | `archive/query/search_hits.py:190` `_hybrid_score_components` → `<lane>_rank`/`<lane>_rrf`+fused; `:216` `primary_lane_evidence`; `:234` `default_score_kind` | already emits per-lane RRF; `_HYBRID_RRF_K=60` duplicated here |
| Hit contract | `search_hits.py:20` `SessionSearchHit`; `storage/search/models.py:32` `SessionSearchEvidenceRow` | carry `score_components`, `score_kind`, `lane_rank`, `lane_contribution`, `raw_score` |
| Public payload | `surfaces/payloads.py:948` `SessionSearchMatchPayload` | already exposes every explain field — **`explain`/`--why` is a rendering concern, not new data plumbing** |
| Session dedup | `storage/search_providers/hybrid_sessions.py:11` `_resolve_ranked_session_hits` | collapses **messages→session** via `ROW_NUMBER() PARTITION BY session_id`; **NO lineage collapse** |
| Lineage collapse key | `session_profiles.logical_session_id` (`index.py:801`, indexed `:875`) = `root_session_id` | exists, unused by search |
| Miss diagnostics | `archive/query/miss_diagnostics.py:25` `QueryMissReason(code,severity,summary,detail,count)` | structural reasons only; **`count` field already present** for relaxation candidates; no relaxation/did-you-mean |
| Assertion substrate | `core/enums.py:399` `AssertionKind` (TEXT col, no CHECK — `user.py:17`); table `assertions` has `scope_ref,target_ref,key,value_json,confidence,evidence_refs_json,context_policy_json` | no `RELEVANCE` kind yet |
| Lane routing | `archive/query/archive_execution.py:362,538` `archive_search_hits(...)→(hits,resolved_lane)` | `auto/dialogue/semantic/hybrid`; no intent classifier upstream |
| devtools lab | `devtools/command_catalog.py` (`lab probe pipeline`, `lab schema *`) | new `lab search-eval` slots here |

---

## (1) Schema / DDL + tier assignment

**Durability principle: human relevance labels are irreplaceable → `user.db`. Everything else is derived → recomputable, no durable schema.**

### 1a. Relevance judgments — `user.db` (durable, tier ver 4, **no DDL migration**)
`AssertionKind.RELEVANCE = "relevance"` added to the closed enum (`core/enums.py:428`). The `assertions.kind` column is `TEXT` with **no CHECK** (`user.py:17`), so **no user-tier `PRAGMA user_version` bump and no numbered migration** — consistent with the CLAUDE.md gotcha for schema-free kinds. Row shape:

```
kind        = "relevance"
target_ref  = "message:<message_id>"  |  "session:<session_id>"   (graded unit)
scope_ref   = "query:<sha256(normalized_query)>"                  (the judged query)
key         = normalized_query_text                               (human-readable)
value_json  = {"grade": 0..3, "query": "...", "lane": "hybrid",
               "config_id": "<ranking-config-hash>", "rank_at_judgment": N}
confidence  = REAL (judge certainty, optional)
author_ref  = "user:local"  (or agent ref for weak/auto labels)
context_policy_json = {"inject": false}   (default; judgments never injected)
```
Graded via existing `mark` verb (`AssertionKind.MARK` path in `user_write.py`) extended with a `--grade` option that switches kind to `RELEVANCE`. Reuses existing dedup/`_deterministic_id` write path.

### 1b. LtR feature corpus + eval snapshots — **derived, `.local/` artifacts (no DB)**
Regression history and the LtR training corpus are recomputable from (judgments × current ranker), exactly like `devtools/benchmark_results.py`. Store as JSONL/JSON under `.local/search-eval/`:
- `ltr-corpus.jsonl` — one row per `(query, hit)`: features `{text_rank, text_rrf, vector_rank, vector_rrf, fused, recency_days, title_match:bool, tool_log_match:bool, variant_count, lane_only:bool}` + `label=grade`.
- `eval-baseline.json` / `eval-<config_id>.json` — per-lane `nDCG@10`, `MRR`, `recall@k` snapshots for gating.

Rationale for **not** using `index.db`: eval output is a report over a rebuildable tier; putting it in `index.db` would couple it to the rebuild-on-mismatch regime for zero benefit. `ops.db` is disposable and wrong for a regression baseline that must survive resets.

### 1c. No FTS/index DDL change for field boost (see §2d) — post-retrieval re-weighting avoids an `index.db` rebuild.

---

## (2) Ranking & eval algorithms (pseudocode)

### 2a. Weighted RRF (generalizes `hybrid.py:47`, back-compat default weights = 1.0)
```
def weighted_rrf(lane_lists: dict[lane, list[(id,_)]], k=60, weights: dict[lane,float]):
    scores = {}
    for lane, results in lane_lists.items():
        w = weights.get(lane, 1.0)
        seen = set()
        for rank, (id, _) in enumerate(results, start=1):
            if id in seen: continue
            seen.add(id)
            scores[id] = scores.get(id, 0.0) + w * (1.0 / (k + rank))
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))   # keep stable tie-break
```
`_HYBRID_RRF_K` in `search_hits.py:187` and `hybrid.py` must reference **one** constant (extract to a shared `ranking_params` module) so decomposition never drifts from fusion.

### 2b. Query-intent classifier (pure fn, pre-routing; no schema)
```
def classify_intent(query, parsed_expr) -> (primary_lane, weights):
    if parsed_expr.has_fielded_id (near:id, session:):      return "navigational", route_direct
    if is_quoted_phrase or has_boolean_ops:                 return "lexical",  {text:1.4, vector:0.6}
    if has_error_tokens("error","traceback","exit code")
       or parsed targets actions unit:                      return "diagnostic",{text:1.0, vector:0.7, action_boost}
    if len(tokens)>=6 and low_rare_term_ratio:              return "conceptual",{text:0.6, vector:1.4}
    return "balanced", {text:1.0, vector:1.0}
```
Feeds `weights` into 2a and picks a `primary_lane` hint for `archive_search_hits` (`archive_execution.py:538`). Classification recorded in explain output.

### 2c. Lineage-aware collapse with `variant_count` (extends `hybrid_sessions.py:11`)
```
-- after messages→session ranking, collapse siblings sharing logical_session_id
ranked AS ( ... existing message_rank per physical session ... )
JOIN session_profiles p ON p.session_id = ranked.session_id
logical AS (
  SELECT COALESCE(p.logical_session_id, ranked.session_id) AS logical_id,
         ranked.session_id, message_rank,
         ROW_NUMBER() OVER (PARTITION BY logical_id ORDER BY message_rank ASC, session_id ASC) AS variant_rank,
         COUNT(*)     OVER (PARTITION BY logical_id) AS variant_count
  FROM ranked ...)
SELECT session_id, variant_count FROM logical WHERE variant_rank = 1
ORDER BY message_rank ASC LIMIT ?
```
Emit `variant_count` onto `SessionSearchIdHit`/`SessionSearchHit`/`SessionSearchMatchPayload`. The archive-execution hybrid path (`archive_execution.py:618`) must apply the **same** collapse — factor it into one `collapse_by_logical_session(rows)` helper used by both resolvers, or the two search paths diverge (a live gotcha this repo already has with fork-composition landing in only some read paths).

### 2d. Field-boosted BM25 (post-retrieval; no FTS rebuild)
FTS is contentless single-column over `blocks.search_text` (`unicode61`, no porter — must not change). So boost **after** retrieval by where the match landed, not via multi-column `bm25()` weights:
```
boost = 1.0
if title_contains_match(hit): boost *= W_TITLE   (e.g. 1.5)
if hit.block.material_origin in {tool_result, runtime_protocol}: boost *= W_TOOL_LOG  (e.g. 0.6)
lane_contribution *= boost   # applied to text lane's rrf before fusion
```
Rejected alt: multi-column weighted FTS (`bm25(fts, w_title, w_body)`) needs an `index.db` FTS schema change + full rebuild for a tunable that post-weighting delivers with a knob.

### 2e. Recency×relevance + MMR + weak_evidence (post-fusion reranker)
```
def rerank(hits, cfg):
    for h in hits:                                   # recency blend
        decay = 0.5 ** (age_days(h) / cfg.half_life_days)
        h.final = (1-cfg.recency_w)*h.fused + cfg.recency_w*decay*h.fused
    selected = mmr(hits, lambda a,b: vector_cos(a,b), lambda_=cfg.mmr_lambda)  # diversity
    for h in hits:                                   # evidence banding
        h.band = "weak" if (single_lane_only(h) and h.lane_rank > cfg.weak_rank_floor) \
                        or h.fused < cfg.weak_floor else "strong"
    return selected
```
MMR needs pairwise session vectors from `embeddings.db` (reuse `vector_provider`); `weak_evidence` band is a label on the hit, surfaced in payload + `--why`.

### 2f. Miss-diagnostic relaxation / did-you-mean (extends `miss_diagnostics.py`)
`QueryMissReason` already carries `count` — reuse it. Add relaxation candidates, each a reason with the **yield count it would produce** (candidate runs are bounded, count-only `SELECT COUNT`):
```
async def relaxation_reasons(plan, repo):
    out = []
    for candidate in [drop_least_frequent_term(plan), or_instead_of_and(plan),
                      widen_date_window(plan), spell_correct(plan.terms),  # trigram/edit-dist vs FTS vocab
                      drop_origin_filter(plan)]:
        n = await repo.count_only(candidate.to_plan())
        if n > 0:
            out.append(QueryMissReason(code=f"relax_{candidate.kind}", severity="info",
                       summary=candidate.describe(), detail=candidate.suggested_query, count=n))
    return sorted(out, key=lambda r: -r.count)[:5]
```
Appended in `diagnose_query_miss` after the existing structural reasons; spell-correct source = FTS `unicode61` term vocabulary (trigram/edit-distance ≤2).

### 2g. Eval metrics (`devtools lab search-eval`)
```
labels = load_relevance_assertions()               # user.db kind=relevance, grouped by scope_ref(query)
for query, graded in labels:
    for lane in [dialogue, semantic, hybrid, hybrid+cfg]:
        ranked = run_lane(query, lane, cfg)
        ndcg[lane]   += ndcg_at_k(ranked, graded, k=10)
        mrr[lane]    += reciprocal_rank(ranked, graded)
        recall[lane] += recall_at_k(ranked, graded, k=cfg.k)
report = mean over queries per lane
gate: FAIL if any(metric[candidate] < metric[baseline] - EPSILON)   # blocks weight/k/boost changes
```
`dcg = Σ (2^grade - 1)/log2(rank+1)`, `nDCG = dcg/ideal_dcg`. Baseline from `.local/search-eval/eval-baseline.json`.

---

## (3) Migration

- **`user.db`**: none. `RELEVANCE` is a schema-free `TEXT` kind (no `PRAGMA user_version` change, no numbered SQL migration). **But regenerate** `render openapi` + `render cli-output-schemas` (enum is embedded there — documented gotcha) and add a `user_audit` surface entry (`user_audit.py:16` map: `"relevance_judgments": AssertionKind.RELEVANCE`) or the every-kind audit invariant fails.
- **`index.db`**: `variant_count` reads the **existing** `session_profiles.logical_session_id` — no new column, no rebuild. Field-boost/recency/MMR/weighted-RRF are compute-only. **No index bump.**
- **New module `polylogue/…/ranking_params.py`** (shared `RRF_K` + `RankingConfig`): adding any `polylogue/` module requires `devtools render topology-projection && devtools render topology-status` or `render all --check` fails.
- **devtools**: new `CommandSpec("lab search-eval", …)` in `command_catalog.py` + `devtools/search_eval.py`; run `devtools render devtools-reference`.
- **MCP**: if a `search --why`/`explain_search` tool is added → update `EXPECTED_TOOL_NAMES` + tool contract or discovery tests fail.

---

## (4) Test strategy

- **RRF invariants (property, Hypothesis)**: weighted_rrf with all weights=1.0 ≡ current `reciprocal_rank_fusion` (regression pin); permutation-invariance of tie-break preserved; `Σ score_components[*_rrf] == fused` (extend `tests/unit/storage/test_search_explanation_wiring.py`).
- **Collapse correctness**: builder with a fork/resume family sharing `logical_session_id` → one hit, `variant_count == N`; both resolvers (`hybrid_sessions` + `archive_execution` hybrid) return identical collapse (guards the divergent-read-path gotcha). Use `SessionBuilder` + `session_links`.
- **Explain contract**: extend `tests/unit/surfaces/test_search_explanation_contract.py` — `--why`/payload narrates each `*_rank`/`*_rrf`, `variant_count`, `band`, intent, field-boost factor; `weak` band surfaces on single-lane low-rank hit.
- **Intent classifier (table-driven)**: quoted/boolean→lexical weights; long conceptual→vector-heavy; error tokens→diagnostic; `near:id`→navigational.
- **Relaxation**: seeded miss (over-constrained plan) yields ordered candidates each with a **real** `count`; count matches an independent `COUNT(*)`; zero-yield candidates suppressed.
- **Eval metrics (unit)**: nDCG/MRR/recall against hand-computed fixtures; gate FAILs on injected regression, PASSes within epsilon.
- **Determinism**: `frozen_clock` for recency; MMR deterministic tie-break.
- Inner loop: `devtools test <file>` / `-k`, never blanket-run. `demo seed` corpus for read/explain checks (private-data-free).

---

## (5) Bead breakdown (acceptance criteria)

1. **`feat: lineage-aware search collapse + variant_count`** — one `collapse_by_logical_session` helper used by both `hybrid_sessions._resolve_ranked_session_hits` and `archive_execution` hybrid path; `variant_count` on hit + payload. **AC**: fork family collapses to 1 hit with correct count in both paths; existing session-dedup tests green; no index bump.
2. **`feat: weighted RRF + shared ranking_params`** — extract single `RRF_K`, add per-lane weights (default 1.0 = byte-identical output). **AC**: property test weights=1.0 ≡ baseline; topology projection regenerated.
3. **`feat: explain search / read --why narration`** — render `score_components`, primary lane, `variant_count`, band, intent, boosts from existing payload fields; add `--why` flag (last positional). **AC**: explanation contract test asserts every field narrated; no new storage plumbing.
4. **`feat: AssertionKind.RELEVANCE + graded mark`** — enum value, `mark --grade`, user_audit map entry, regenerated openapi/cli-output-schemas. **AC**: grade round-trips to `user.db`; `render all --check` clean (grep `out of sync`); audit invariant holds.
5. **`feat: devtools lab search-eval (nDCG@10/MRR/recall + gate)`** — eval over RELEVANCE corpus, `.local/search-eval/` baseline, regression gate. **AC**: metrics match fixtures; gate FAILs on injected regression; `render devtools-reference` updated.
6. **`feat: query-intent classifier + recency×relevance/MMR/weak_evidence knobs`** — `RankingConfig`, classifier pre-routing, post-fusion reranker. **AC**: classifier table tests pass; band label surfaces; knobs config-driven, defaults preserve current ranking; gated by bead 5's eval (no nDCG regression).
7. **`feat: field-boosted bm25 (post-retrieval)`** — title/tool-log weighting on text lane before fusion, no FTS rebuild. **AC**: title-match hit outranks equal body-match; eval nDCG non-regressing.
8. **`feat: miss-diagnostic relaxation + did-you-mean`** — bounded count-only candidates in `diagnose_query_miss`, ordered by yield. **AC**: over-constrained miss returns candidates each with verified `count`; spell-correct from FTS vocab; CLI + MCP surfaces render them.

Suggested order: 1→2→3 (explainability spine), 4→5 (eval substrate), then 6/7 gated on 5, 8 independent. Beads 6/7 must not merge without a passing `lab search-eval` gate.

---

## (6) Top-3 risks

1. **Divergent read paths (highest).** Session resolution happens in *two* places (`hybrid_sessions._resolve_ranked_session_hits` and `archive_execution.py:618` hybrid). This repo already shipped fork-composition into only some paths (memory: #2470). Collapse/`variant_count` applied to one but not the other silently returns duplicate lineage siblings on the untouched path. **Mitigate**: single shared collapse helper + a test asserting both paths agree, before any ranking work lands.
2. **Eval corpus cold-start / construct validity.** nDCG/MRR gating is only meaningful with enough RELEVANCE labels; a thin, self-selected corpus makes the gate either vacuous or actively misleading (over-fits ranker to a handful of pet queries). **Mitigate**: gate reports label coverage and refuses to fail-block below a minimum judged-query count; weak/auto labels (agent-authored, lower `confidence`) kept separable from human grades so they can't silently dominate.
3. **Ranking-config drift vs explanation.** `_HYBRID_RRF_K` is duplicated across `hybrid.py` and `search_hits.py`; adding weights/boosts/recency multiplies the surfaces where fusion math and the `score_components` decomposition can silently disagree — making `--why` lie. **Mitigate**: one `ranking_params`/`RankingConfig` source of truth, and a property test pinning `Σ *_rrf (post-weight/boost) == fused` so the explanation is provably the arithmetic that ran.

**No code or beads were modified.**

---

## [a0e82ce27ad04636f] Spec — daemon / capture security hardening

I have the full picture. Returning the spec.

---

# Spec — daemon / capture security hardening

Grounded in `daemon/http.py`, `browser_capture/{server,receiver}.py`, `core/loopback.py`, `docs/daemon-threat-model.md`, bead **polylogue-kwsb** (security/privacy epic; its design note explicitly says *"Runtime controls exist and are tested (http.py auth/CSRF, MCP role contracts) but the BACKLOG ownership was missing"* — this work is a new child of kwsb).

## 0. Confirmed defect map (file:line)

| # | Defect | Evidence |
|---|--------|----------|
| D1 | **No Host check on any route; Origin checked only on POST and skipped when absent** → DNS-rebinding reads whole archive via GET. | GET dispatch `_dispatch_get` (http.py:1259) never inspects `Host`. `_check_cross_origin` (1369) is POST-only and returns `True` on missing Origin (1376). |
| D2 | **Query-token fallback on *all* routes**, not just SSE. | `_check_auth` (1105) applies `?access_token=` fallback (1117-1122) for every request. Widens token leakage (referer logs, shell history, proxy logs). |
| D3 | **Non-constant-time token compare** in both surfaces. | http.py `_check_auth_logic` (1061): `auth_header[7:] != auth_token`. server.py `_check_token` (73): `auth[7:] == config.auth_token`. |
| D4 | **Receiver has no auth on loopback by default** → any local process `curl`-POSTs forged captures into the spool (Origin absent ⇒ `_origin_allowed(None)` returns True, receiver.py server.py:54-61; token None ⇒ `_check_token` returns True, 68-73). | `BrowserCaptureReceiverConfig.auth_token` defaults `None` (receiver.py:52). |
| D5 | **Spool disk-fill DoS.** 128 MiB/request cap (server.py:47) but unbounded request count; attacker-varied `provider_session_id` defeats the dedup path (`_safe_token`, receiver.py:104) → unbounded files. No per-origin rate limit, no aggregate spool cap. | `write_capture_envelope` (receiver.py:318) writes unconditionally. |
| D6 | **`/metrics` unauthenticated even off-loopback; OTLP POST unauthenticated on loopback and bypasses the Origin/CSRF gate.** | `/metrics` (http.py:1312) served before `_check_auth`. OTLP route returns early (1406-1422) *before* `_check_cross_origin` (1426); auth only required when `not is_loopback_host` (1415). ⇒ rebinding/CSRF write into OTLP + unauth metrics scrape. |

The unifying gap: **loopback binding conflates two trust classes the network layer cannot separate** — the daemon's own web shell (same-origin, trusted) and a browser page on another origin that resolved a hostname to 127.0.0.1 (untrusted). Host/Origin is the only signal that discriminates them.

## 1. Trust-class model (the invariant middleware enforces)

```
T0  loopback same-user process (curl, another CLI) — ambient FS authority, can read token files
T1  daemon's own web shell / same-origin fetch      — MUST keep working
T2  browser page from a foreign origin (rebinding/CSRF) — UNTRUSTED despite hitting 127.0.0.1
T3  remote network                                   — blocked by loopback bind
```

- Bind (T3) is already handled. **Host+Origin allowlist is what separates T1 from T2.** Token separates T0 from unauthorized T0.
- Recursive-safety: the gate is applied **once, centrally, before dispatch**, so every current and future route inherits it (no per-handler opt-in that a new route can forget — the class of D1/D6 bugs).

## 2. Middleware + schema/config changes

### 2a. `core/loopback.py` — add a Host-allowlist primitive (shared substrate)

```
DEFAULT_ALLOWED_HOST_NAMES = {"localhost", "127.0.0.1", "[::1]", "::1"}

def host_header_allowed(host_header: str, allowed: frozenset[str], bind_port: int|None) -> bool:
    # host_header is the raw "Host:" value: "name[:port]" or "[v6]:port"
    if not host_header: return False           # HTTP/1.1 requires Host; absence = reject
    name, port = split_host_port(host_header)  # bracket-aware, reuse is_loopback_origin parsing
    if port is not None and bind_port is not None and port != bind_port: return False
    name = name.strip().lower()
    return name in allowed or is_loopback_host(name)
```

Reuse the bracket/`:port` parsing already proven in `is_loopback_origin` (loopback.py:58-87) — do not hand-roll a second parser.

### 2b. Daemon config (`config.py`) + server

- New config field `api_allowed_hosts: tuple[str, ...] = ()` (5-layer resolution like `api_host`). Empty ⇒ derive `DEFAULT_ALLOWED_HOST_NAMES ∪ {api_host}`. Repeatable `--api-allowed-host` CLI flag for reverse-proxy / custom-hostname deployments (the only legitimate `Host != loopback` case).
- `DaemonAPIHTTPServer.__init__` (http.py:3877) gains `allowed_hosts: frozenset[str]` and stores it; handler reads via a property mirroring `_api_host` (1096).

### 2c. `daemon/http.py` — central Host gate + narrowed token fallback

- Add `_check_host() -> bool` and call it as the **first line of both `do_GET` and `_do_post_impl`**, before web-shell/healthz/metrics/OTLP branches. This closes D1 and D6 in one place.
- `_check_auth(self, *, allow_query_token: bool = False)`: move the `?access_token=` block (1117-1122) behind `allow_query_token`. Only `_handle_events` (SSE, 1748) passes `True`; wire it by having the events route call `self._check_auth(allow_query_token=True)` explicitly rather than the generic pre-dispatch `_check_auth()`. Closes D2.
- Replace `!=` at `_check_auth_logic` (1061) with `not hmac.compare_digest(auth_header[7:], auth_token)`. Closes D3 (daemon half).
- `/metrics`: require `_check_host()` (already now central) **and** `_check_auth()` when `not is_loopback_host(self._api_host)` — mirror the OTLP posture so a non-loopback deploy doesn't leak metrics. OTLP route: move it **after** `_check_host()`; keep the loopback-auth rule but it now also can't be reached cross-origin.

### 2d. `browser_capture` — token on loopback, constant-time, Host gate

- `_check_token` (server.py:73): `hmac.compare_digest(auth[7:], config.auth_token)`. Closes D3 (receiver half).
- Add a `_check_host` reject to `_do_get`/`_do_post`/`_do_options` (before `_reject_origin`), using the same `host_header_allowed` primitive with the receiver's bind host+port.
- **Auto-provisioned loopback token (D4).** `BrowserCaptureReceiverConfig` gains factory behavior: when `auth_token is None`, `make_server` (server.py:385) generates `secrets.token_urlsafe(32)` and writes it to `spool/.receiver-token` mode `0600`. The daemon's browser-capture bootstrap (`daemon/browser_capture.py`) injects that token into the extension's config handshake. `_reject_token` (server.py:162) now fires even on loopback. Same-user processes can still read the file (in-scope residual per threat model §Archive file tampering), but confused-deputy browser pages and other-user/sandboxed processes cannot forge. `validate()` (receiver.py:58) drops the "loopback ⇒ token optional" implication; token is always present post-provision.

### 2e. `browser_capture/receiver.py` — spool governor (D5)

New `SpoolGovernor` consulted by `write_capture_envelope` before the atomic write:
- **Size cap:** aggregate spool bytes ≤ `POLYLOGUE_BROWSER_CAPTURE_SPOOL_MAX_BYTES` (default e.g. 2 GiB) and file count ≤ N; over cap ⇒ `SpoolQuotaExceededError` → `507 Insufficient Storage`.
- **Per-origin token bucket:** key = `Origin` header value (or `"local"` when absent); refill R captures/min, burst B; over budget ⇒ `429 Too Many Requests` with `Retry-After`.

## 3. Hardening algorithms (pseudocode)

```
# Central request gate — runs before ANY dispatch (both surfaces)
def guard(request):
    if not host_header_allowed(request.header("Host"), allowed_hosts, bind_port):
        return 403 "host_not_allowed"          # closes DNS-rebinding for GET and POST
    # ... surface-specific origin/token below

# Daemon auth (constant-time; query-token only for SSE)
def check_auth(request, allow_query_token=False):
    if not configured_token: return ALLOW      # open-loopback dev mode unchanged
    header = request.header("Authorization")
    if not header and allow_query_token:
        header = "Bearer " + request.query["access_token"]
    if not header.startswith("Bearer "): return DENY(401)
    return ALLOW if hmac.compare_digest(header[7:], configured_token) else DENY(401)

# Receiver spool admission
def admit_capture(origin, envelope_bytes):
    if spool_total_bytes() + len(envelope_bytes) > SPOOL_MAX_BYTES: raise 507
    if spool_file_count() >= SPOOL_MAX_FILES:                       raise 507
    bucket = buckets[origin or "local"]
    bucket.refill(now)                          # tokens += elapsed * R, capped at B
    if bucket.tokens < 1: raise 429(Retry-After = ceil((1-bucket.tokens)/R))
    bucket.tokens -= 1
```

## 4. Migration

**No DB-tier migration** — index/source/user/embeddings/ops schemas are untouched; this is runtime + config only. Steps:

1. **Config additive**, backward-compatible: `api_allowed_hosts` empty ⇒ auto-derive; existing `--api-host`/`--api-auth-token` semantics unchanged. Existing loopback deployments keep working with no operator action.
2. **Receiver token becomes mandatory-but-auto**: on first start after upgrade the receiver mints `.receiver-token`; the extension picks it up through the existing status handshake. Operators who already set `--browser-capture-auth-token` are unaffected. Document the file (`0600`) and its regeneration semantics in `docs/security.md`.
3. **Regenerate generated surfaces**: `--api-allowed-host` is a new Click param on `polylogued run` → `devtools render cli-reference` + CLI help snapshots (`tests/unit/cli/__snapshots__/…`). No topology projection change (no new module if governor lives in `receiver.py`).
4. **Threat-model doc**: rewrite `docs/daemon-threat-model.md` §Trust Boundary + the "Local process reading the API" / "Extension posting forged captures" threat rows to state Host-allowlist and receiver-token as *mitigations now present*, not residuals. Add the trust-class table.
5. **Backout**: purely additive; reverting the commit restores prior behavior with no data implication.

## 5. Test strategy

Layer under `tests/unit/daemon/` and `tests/unit/browser_capture/`; extend `tests/unit/security/` (a protected dir — additions welcome).

- **Rebinding-attack fixture** (the keystone). A pytest fixture `rebinding_client` that drives the real `DaemonAPIHTTPServer`/`BrowserCaptureHTTPServer` on an ephemeral loopback port but sends a **forged `Host: attacker.example`** (and, in a second variant, `Host: 127.0.0.1` with `Origin: https://attacker.example`) on GET `/api/sessions`, `/api/sessions/:id`, `/metrics`, and POST `/v1/traces`, `/v1/browser-captures`.
  - Assert **403 `host_not_allowed`** for the foreign-Host variant on every route (proves D1/D6 closed and that the gate is recursive across the whole route table — enumerate `implemented_daemon_route_patterns()` (http.py:359) so a newly-added route without the gate fails the test).
  - Assert the **same-origin web shell still works**: `Host: 127.0.0.1:<port>` + `Origin: http://127.0.0.1:<port>` GET `/` and `/api/sessions` → 200 (proves T1 not broken).
- **Query-token scoping (D2):** `?access_token=` on `/api/sessions` → 401; on `/api/events` → 200. Parametrize over the route table asserting only the SSE route honors it.
- **Constant-time (D3):** unit-test `_check_auth_logic` / `_check_token` route through `hmac.compare_digest` (assert the call, or property-test that wrong tokens of varying prefix-match length all reject — a behavioral not timing assertion; a real timing test is flaky and out of scope).
- **Receiver token on loopback (D4):** default-config receiver, no `Authorization` header, loopback client → 401 (regression: was 202). With the minted token → 202. Assert `.receiver-token` is `0600`.
- **Spool governor (D5):** feed N+1 captures with distinct `provider_session_id`; assert 429 after the burst and 507 once the byte/file cap trips; assert `Retry-After` present. Use `frozen_clock` (`tests/infra/frozen_clock.py`, per CLAUDE.md clock hygiene) to drive token-bucket refill deterministically.
- **Verification command:** `devtools test tests/unit/daemon/test_http_host_gate.py tests/unit/browser_capture/test_receiver_auth.py tests/unit/security/` then `devtools verify` (testmon-affected) + `devtools render all --check` (grep for `out of sync`, per the known false-`sync OK` gotcha).

## 6. Bead breakdown (children of polylogue-kwsb)

1. **`host-allowlist-middleware`** — `host_header_allowed` in `core/loopback.py` + central `_check_host()` first-line gate in daemon `do_GET`/`_do_post_impl` + receiver handlers; `api_allowed_hosts` config + `--api-allowed-host`.
   *AC:* forged-`Host` GET/POST → 403 on **every** route in `implemented_daemon_route_patterns()`; same-origin loopback web shell → 200; `devtools render cli-reference` regenerated. Rebinding fixture green.
2. **`constant-time-token-compare`** — `hmac.compare_digest` in http.py `_check_auth_logic` and server.py `_check_token`.
   *AC:* both compares route through `compare_digest`; wrong-token variants (all prefix lengths) reject; no behavior change on correct token.
3. **`scope-query-token-to-sse`** — `_check_auth(allow_query_token=)`; only `_handle_events` opts in.
   *AC:* `?access_token=` → 401 on non-SSE routes, 200 on `/api/events`; parametrized across route table.
4. **`receiver-loopback-token`** — auto-mint `secrets.token_urlsafe(32)` → `spool/.receiver-token` (0600) in `make_server`; `_reject_token` fires on loopback; extension handshake injection; `validate()` updated.
   *AC:* unauthenticated loopback POST `/v1/browser-captures` → 401; minted-token POST → 202; token file `0600`; extension bootstrap carries the token.
5. **`spool-governor`** — `SpoolGovernor` (per-origin token bucket + aggregate byte/file cap) consulted in `write_capture_envelope`.
   *AC:* burst → 429 w/ `Retry-After`; cap → 507; `frozen_clock`-driven refill test; distinct-session flood no longer unbounded.
6. **`metrics-otlp-posture`** — `/metrics` requires auth when non-loopback; OTLP POST moved behind `_check_host`.
   *AC:* non-loopback `/metrics` without token → 401; cross-origin OTLP POST → 403; loopback OTLP unchanged (still gated on `observability_enabled`).
7. **`threat-model-doc-refresh`** — rewrite `docs/daemon-threat-model.md` (trust-class table; Host-allowlist + receiver-token as present mitigations) + `docs/security.md`; note at-rest plaintext remains OS/disk-encryption responsibility (documented non-goal, links kwsb→27m excision for the "forget a pasted secret" half).
   *AC:* doc lists each mitigation with its enforcing `file:function`; `render all --check` clean.

Sequencing: #1 is the spine (others depend on the central gate existing); #2/#3 are independent small; #4→#5 ordered (governor sits behind the same handler path); #6 depends on #1; #7 last, after code lands.

## 7. Top-3 risks

1. **Breaking the same-origin web shell / SSE.** The Host allowlist must admit every legitimate `Host` the shell emits: bare `127.0.0.1`, `localhost`, `[::1]`, *with and without the bound port*, plus any operator `--api-host`. A too-strict port match breaks reverse-proxy/port-forward setups; a too-loose name match reopens rebinding. Mitigation: derive the allowlist from bind config, port-match only when a port is present, and gate the whole thing behind the rebinding fixture asserting **both** the block and the shell-still-works directions.
2. **Receiver token distribution to the extension.** Auto-minting is only useful if the extension actually receives the token; the handshake path (`daemon/browser_capture.py` → extension config) is the fragile seam. If injection fails, captures silently 401 and the dogfood capture loop breaks. Mitigation: an end-to-end test through the real handshake, and a startup log line asserting the token was surfaced; keep `--browser-capture-auth-token` as the explicit override.
3. **Same-user residual is unchanged and must not be over-claimed.** The receiver token file and the archive SQLite are readable by any same-user process — the threat model's declared out-of-scope boundary. These changes stop *confused-deputy browser pages* and *rebinding*, not a hostile same-user process. Risk is a doc/PR overclaim ("receiver is now authenticated") that a future audit falsifies. Mitigation: threat-model bead (#7) states the residual explicitly; PR body frames this as defense-in-depth against the browser/network trust classes (T2), consistent with kwsb's sibling jsy framing (*"none currently exploitable; defense-in-depth"*).

---

## [a3181e3969bdc79cf] SPEC: cross-project agent-memory `recall(task_hint)` MCP tool

I now have complete ground truth. Writing the spec.

---

# SPEC: cross-project agent-memory `recall(task_hint)` MCP tool

Grounded in the live tree. The load-bearing discovery: **`VectorProvider.query(text, limit) -> [(message_id, distance)]`** (`polylogue/protocols.py:59`, impl `polylogue.storage.search_providers.sqlite_vec.SqliteVecProvider`, factory `_create_vector_provider` in `polylogue/cli/query.py`) already embeds arbitrary text via Voyage (query input-type) and KNN-searches `message_embeddings`. This is the exact query→vector primitive `recall(task_hint)` needs — the existing `find_similar_sessions`/`neighbor_candidates`/`daemon/similarity.py:build_similar_payload` are all **session→session** (require the seed to already have stored vectors); none embeds a free-text hint. `recall` is the missing text-anchored, cross-repo, evidence-joining leg. It is the MCP-invoked, cross-project sibling of bead **mhx.4** (semantic recall in context compilation) and registers as a ContextSource under **37t.11** in the eventual scheduler world; as a standalone MCP tool it serves the mid-session "recall the last time we hit this" moment.

## 1. MCP tool contract + schema

**Home:** `polylogue/mcp/server_insight_tools.py` (read-shaped, delegates to a new `Polylogue.recall_report(...)` façade method; mirrors `get_pathologies`/`find_similar_sessions` exactly). Registered by the `@mcp.tool()` decorator; add name to `EXPECTED_TOOL_NAMES` (`tests/infra/mcp.py:15`) and a tool-contract row (precedent: recent `agent_coordination` addition).

```python
@mcp.tool()
async def recall(
    task_hint: str,                       # free-text description of current task; embedded as the recall anchor
    repos: str | None = None,             # CSV include-filter of repo scopes; None = ALL repos (cross-project default)
    exclude_repos: str | None = None,     # CSV exclude-filter (e.g. drop the current repo to force cross-project)
    kinds: str | None = "lesson,blocker,correction,caveat,pathology",  # evidence kinds to attach
    session_limit: int = 6,               # top-N similar prior sessions
    token_budget: int = 1200,             # hard cap on assembled payload token cost
    min_similarity: float = 0.35,         # cosine floor; hits below are dropped, never padded
    statuses: str | None = "active",      # assertion lifecycle floor; "active,candidate" to see unjudged (QUOTED)
) -> str: ...
```

**Response schema** (`MCPRecallPayload`, new Pydantic model in `polylogue/mcp/payloads.py`, `exclude_none=True`):

```jsonc
{
  "task_hint": "…",
  "retrieval_lane": "vector" | "fts",        // HONEST degradation flag (never silent)
  "embeddings_status": "ready|disabled|unavailable|not_embedded",
  "reason": null | "no_voyage_api_key" | "sqlite_vec_not_loaded" | ...,
  "scope": { "included_repos": [...], "excluded_repos": [...], "cross_project": true },
  "token_budget": 1200,
  "tokens_used": 1043,
  "sessions": [
    { "session_id": "...", "repo": "sinex", "origin": "codex-session",
      "title": "...", "similarity": 0.71, "matched_messages": 4,
      "resolve_ref": "session:...", "why": ["cosine 0.71 over 4 messages"] }
  ],
  "evidence": [
    { "kind": "correction", "trust_class": "OPERATOR",
      "text": "…", "author_ref": "operator", "judged": true,
      "scope_ref": "repo:sinex", "resolve_ref": "assertion:...",
      "why": {"similarity": 0.71, "kind": "correction", "judgment": "active"} },
    { "kind": "lesson", "trust_class": "QUOTED",
      "quoted_text": "«reported by agent in session … — UNVERIFIED»\n> …",
      "author_ref": "agent:...", "judged": false, "resolve_ref": "assertion:..." },
    { "kind": "pathology", "trust_class": "SYSTEM",
      "detail": "wasted-loop x3", "resolve_ref": "insight:pathology:...",
      "evidence_refs": ["message:...", "message:..."], "text": null }
  ],
  "degraded": [ {"resolve_ref": "assertion:...", "reason": "over_budget"} ]  // dropped/ref-only items
}
```

**Recursive-safety gate (trust class per item)** — the non-negotiable core, from the 37t.11 injected-context trust doctrine:
- `OPERATOR` — assertion is `status=active` **and** human-authored/judged (`author_ref` = operator, or a candidate promoted via the judgment queue). May be rendered as directive prose.
- `QUOTED` — any agent-authored or `candidate`/unjudged claim (lesson/blocker/correction reported by a prior agent). **Always** fenced, attributed, and framed "reported by … — unverified"; never emitted as an instruction. This is the laundering barrier: an agent's own past claim cannot re-enter another agent's context as authoritative.
- `SYSTEM` — deterministic structural facts (pathology findings, session metadata, similarity scores). Ref + short detail only; **no verbatim stored prose**.
- **No verbatim `tool_result`/web text at any class** — recalled evidence of that kind is `resolve_ref`-only. Class is a property of the *source*, not a per-call flag.

## 2. Recall algorithm (pseudocode)

```
recall(task_hint, repos, exclude_repos, kinds, session_limit, token_budget, min_similarity, statuses):

  # --- Stage A: anchor retrieval (text -> sessions), cross-repo by construction ---
  provider = create_vector_provider(config, embeddings.db)     # None if VOYAGE key absent / vec unavailable
  if provider is not None:
      raw = provider.query(task_hint, limit=session_limit * PER_MESSAGE_FANOUT)   # [(message_id, L2_distance)]
      lane = "vector"
  else:
      raw = fts_over_sessions(task_hint, limit=…)               # honest fallback
      lane = "fts"; record embeddings_status/reason

  by_session = aggregate_best(raw)          # reuse daemon/similarity._aggregate_hits + _l2_to_cosine_similarity
  sessions = [s for s in by_session if cosine(s) >= min_similarity]
  sessions = apply_repo_scope(sessions, repos, exclude_repos)   # resolve each session's repo; include/exclude
  sessions = top_n(sessions, session_limit)                     # rank by cosine, then recency, then id

  # --- Stage B: evidence join (per recalled session's repo + direct target) ---
  scopes = {f"repo:{s.repo}" for s in sessions}
  assertions = list_assertion_claim_payloads(                   # storage/repository assertion reader
      kinds=parse(kinds) ∩ assertion_kinds,                     # lesson/blocker/correction/caveat
      scope_ref ∈ scopes  OR  target_ref ∈ session_ids,
      statuses=parse(statuses))
  pathologies = pathology_report(spec_for(session_ids), limit)  # deterministic, if "pathology" in kinds

  # --- Stage C: trust classification (recursive-safety) ---
  items = []
  for a in assertions:
      cls = OPERATOR if (a.status == active and human_authored(a)) else QUOTED
      items.append(classify(a, cls))            # QUOTED => fence + attribute + "unverified"
  for p in pathologies:
      items.append(classify(p, SYSTEM))         # ref + detail only, evidence_refs, no prose

  # --- Stage D: relevance scoring + token-budgeted assembly ---
  for it in items: it.score = w_sim*sess_similarity(it) + w_kind*kind_weight(it) + w_judge*judged(it)
  items.sort(desc score); ordered greedily:
      for it in items:
          cost = token_cost(it, refs_over_bodies=(it.body_tokens > REF_THRESHOLD))
          if tokens_used + cost <= token_budget: include(it); tokens_used += cost
          else: degrade(it -> ref_only) or drop -> degraded[]
      # invariant: assembled payload never exceeds token_budget

  record_recall_run(task_hint, lane, session_ids, tokens_used)  # ops.db (disposable) — eval/uplift evidence
  return MCPRecallPayload(...)
```

Determinism: identical archive + hint ⇒ identical payload (property-testable), because vector distances, detectors, and greedy fill are all deterministic.

## 3. Migration

- **No durable-tier migration.** Primary lane reuses the existing `message_embeddings` vec0 table (`index.db`/`embeddings.db`, schema v1) with zero DDL change — `VectorProvider.query` already reads it.
- **user.db: unchanged.** Assertions are already the unified `assertions` table; `AssertionKind` values (lesson/blocker/correction/caveat/pathology) exist; `scope_ref` already carries `repo:<name>`. No new kind, no CHECK edit.
- **Assertion-by-meaning (follow-up, derived tier):** to rank *assertions themselves* by embedding rather than by scope-join, add a derived `assertion_embeddings` vec0 table to `embeddings.db`. This is a **rebuild, not a migration** (derived tier rule): bump `embeddings.py:EMBEDDING_SCHEMA=1→2`, edit the canonical DDL + rebuild plan, and reconverge via `polylogue ops reset --embeddings && polylogued run` — never an upgrade helper (`devtools lab policy schema-versioning` rejects one). Gate behind its own bead so v1 `recall` ships without it (scope-join + FTS-over-assertions covers the assertion leg).
- **ops.db (disposable):** additive `recall_runs` table for eval/uplift instrumentation (or reuse `daemon_events`). Disposable tier → free to add.
- **Generated surfaces:** `EXPECTED_TOOL_NAMES += "recall"`; new tool-contract row; `devtools render openapi` + `render cli-output-schemas` (new payload model/enums); new module under `mcp/` (if split out) ⇒ `devtools render topology-projection && render topology-status` or `render all --check` fails.

## 4. Test strategy

- **Trust-class red-team (protected-grade):** seed a session/assertion whose body contains an injection string ("ignore prior instructions…"). Assert the assembled `recall` payload never emits it as OPERATOR/unfenced; agent-authored claims land as QUOTED with attribution + "unverified"; no verbatim `tool_result` text appears at any class. Mirrors the 37t.11 red-team fixture requirement.
- **Cross-project scope:** seed sessions+assertions across ≥3 repos; assert default (no `repos`) recalls across all; `exclude_repos=<current>` forces cross-project; `repos=a,b` restricts. Assert each recalled session's `repo` is resolved correctly.
- **Budget invariant (property/Hypothesis):** for arbitrary item sets and `token_budget`, `tokens_used <= token_budget` always; over-budget items appear in `degraded[]`, never silently dropped without a reason.
- **Determinism (property):** same seeded archive + `task_hint` ⇒ byte-identical payload.
- **Degradation honesty:** with `embedding_enabled=false`/no Voyage key, `retrieval_lane="fts"` + populated `embeddings_status`/`reason`; never a silent empty result (matches `build_similar_payload` contract).
- **Relevance floor:** hits below `min_similarity` excluded; empty recall returns an explicit empty envelope, not fabricated neighbors.
- **Discovery contracts:** `test_tool_discovery` / `EXPECTED_TOOL_NAMES` / tool-contract + envelope-schema tests pass.
- Use `frozen_clock`; seed via `SessionBuilder`/`corpus_seeded_db`; demo-path (`polylogue demo seed`) smoke for a private-data-free end-to-end.

## 5. Bead breakdown (child of s7ae / sibling of mhx.4)

1. **`recall` façade + vector anchor lane (size:M).** `Polylogue.recall_report()` embedding `task_hint` via `VectorProvider.query`, session aggregation, cross-repo scope resolution. **AC:** given a hint, returns ranked cross-repo sessions with cosine ≥ floor + resolve_refs; FTS fallback sets `retrieval_lane`; determinism property test passes.
2. **Evidence join: lessons/blockers/corrections/pathologies (size:M).** Attach assertions (scope_ref/target join) + `pathology_report`. **AC:** each recalled session's active assertions + deterministic pathology findings appear with resolve_refs + evidence_refs; kinds filter honored.
3. **Recursive-safety trust classifier (size:M, blocking).** OPERATOR/QUOTED/SYSTEM per item; QUOTED fencing+attribution; no verbatim tool_result/web. **AC:** red-team fixture never yields unfenced QUOTED or directive-framed agent claim; injection-string fixture never reaches OPERATOR class.
4. **Token-budgeted assembly + refs-over-bodies (size:S).** Greedy scored fill, degrade ladder, `degraded[]`. **AC:** budget-invariant property test; over-budget items ref-only/dropped-with-reason.
5. **MCP tool wiring + contracts (size:S).** `@mcp.tool() recall`, `MCPRecallPayload`, `EXPECTED_TOOL_NAMES`, tool contract, openapi/cli-output-schema render, topology projection. **AC:** discovery + envelope-contract tests green; `render all --check` clean.
6. **ops.db recall-run instrumentation (size:S).** Additive `recall_runs`; feeds mhx.4 eval/uplift. **AC:** every `recall` invocation logs a queryable row; disposable-tier rebuild-safe.
7. *(follow-up)* **`assertion_embeddings` vec0 lane (size:M, derived rebuild).** Embed assertion bodies; recall assertions by meaning not just scope-join. **AC:** embeddings.db v2 via rebuild plan (no upgrade helper); assertion recall lane measurable in eval set.
8. *(follow-up)* **Register `recall` as a ContextSource under 37t.11 (size:M).** Mid-session on-demand moment; scheduler owns budget/ledger/dedup. **AC:** recall items carry trust class into the ledger; budget arbitration delegated to the scheduler.

## 6. Top-3 risks

1. **Query embedding requires the Voyage provider online.** Unlike session→session similarity (reads stored vectors, no key), embedding a free-text `task_hint` needs a live `VOYAGE_API_KEY` call. Mitigation: `retrieval_lane="fts"` degradation is first-class and honest — never silently return zero or pretend. Ranking quality drops on the FTS lane; document it in the payload, don't hide it.
2. **Trust-class laundering is the whole point and easy to get wrong.** If a candidate/agent-authored lesson is ever classified OPERATOR (or QUOTED text is emitted unfenced), `recall` becomes a channel for one agent to inject unverified claims into another's context as authority — a prompt-injection amplifier across the whole corpus. Mitigation: class derived structurally from source (author + judgment state), red-team fixture gates CI, `SYSTEM` items are prose-free by construction. Build it in from the first slice (bead 3 blocking) — retrofitting trust classes is how injection holes ship.
3. **Cross-repo scope + relevance dilution.** Defaulting to ALL repos maximizes "recall the last time we hit this" reach but risks flooding results with high-similarity-but-wrong-repo noise, or leaking sensitive cross-project content into an unrelated session. Mitigation: `min_similarity` floor, `exclude_repos` to drop the caller's own repo when they want *other* contexts, repo shown per session for auditability, and token budget bounding total exposure. Repo resolution correctness (session → repo scope) is a sharp edge — a mis-resolved repo silently mis-scopes evidence.

---

**Key file anchors:** `polylogue/protocols.py:59` (`VectorProvider.query`), `polylogue/daemon/similarity.py` (`_aggregate_hits`, `_l2_to_cosine_similarity`, degradation-status vocabulary to reuse), `polylogue/mcp/server_insight_tools.py:653/1144/386` (`find_similar_sessions`/`get_pathologies`/`get_resume_brief` patterns to mirror), `polylogue/mcp/server_tools.py:737` (`list_assertion_claims` → `list_assertion_claim_payloads(kinds, scope_ref, target_ref, statuses, context_inject, limit)`), `polylogue/archive/query/expression.py:2582` (`repo:` scope pushdown), `tests/infra/mcp.py:15` (`EXPECTED_TOOL_NAMES`), beads **mhx.4** (semantic recall — same retrieval moment, non-MCP), **37t.11** (ContextSource trust doctrine + ledger), **s7ae** (coordination substrate parent).

---

## [a81f80c16d4d591ce] Polylogue: whole-product + launch staircase

I have enough grounding. Here is the synthesis.

---

# Polylogue: whole-product + launch staircase

Grounding checked: `bd show polylogue-3tl` (epic already reframed to "system of record for AI work" + four verbs; 3/18 children done), `release*.yml`/`container.yml`/`homebrew-bump.yml`/`flakehub.yml` all wired but **zero git tags, version frozen `0.1.0`**, `docs/installation.md` + `docs/getting-started.md` both honestly say "no packaged install path until release artifacts exist and smoke-tested," raw-log lines 2016/2134/2183/2192/2233 (operator's true goal = ambient personal utility + agent session-recovery, *not* revenue; "if by the end I am not eagerly using it myself all the time, something is deeply wrong"), and `polylogue-cfk` (the uplift re-run — currently the only recorded experiment says **"packs lose"**, which gates every memory claim).

---

## 1. Identity decision: OSS-by-necessity personal substrate. Not a product.

**Decision: ship it as a serious open-source *tool*, positioned as the operator's own system of record, with the code and findings public. Do not adopt product identity — no "users," no roadmap-as-promise, no support surface, no signup, no "getting started for teams."**

Why this and not product:

- **The operator's stated success condition is personal-use eagerness, not adoption** (raw-log 2129, 2134). A product identity would distort every downstream decision toward a stranger's onboarding funnel and away from the dogfood loop that is the actual point. Building for imaginary users is the classic way solo infrastructure dies.
- **The value that a stranger judges competence on is already the honest one**: a rigorously-modelled substrate (5-tier durability-keyed schema, lineage normalization, contentless FTS, generated-column identity, provider-reported outcomes never regex-guessed). That reads as *engineering seriousness*, which is exactly the fundable/resume-legitimacy signal — and it only reads that way if it is presented as "here is a hard problem I actually solved for myself," not "here is a startup." Competence judgments reward the former and discount the latter.
- **OSS is *by necessity*, not strategy**: it is a local-first, single-writer, no-server tool that reads files you already have. There is no SaaS to sell without betraying the local-first invariant (README: "no upload, no API key"). Closed-source would forfeit the only distribution that fits — inspection.
- The epic already names the category correctly ("the system of record for AI work") and warns against borrowing chat-viewer/observability/memory/QS buckets. Product-identity *is* one of those borrowed buckets. Hold the line the epic already drew.

**What "OSS-by-necessity personal substrate" commits you to:** a real README, a real one-command demo, real published findings on your own archive, a *proven* install path — and nothing that implies a vendor relationship. The repo is the artifact. The findings are the proof. The install works. That is the whole obligation.

---

## 2. The whole-product wrapper, ranked by marginal competence-signal per unit effort

A stranger forms a competence judgment in ~90 seconds on the GitHub page and ~5 minutes if they run anything. Rank by *that* judgment per unit of remaining effort. Most of the stack is already built; the leverage is now in **proving and reconciling**, not building.

**BUILD (in order):**

1. **Release cutover — fire the first tag. (Highest leverage, near-zero build effort.)** Every install doc says "not available yet"; the machinery to make it available is fully wired and idle. Merging the release-please PR converts the single largest credibility gap (README promises a tool a stranger cannot install) into a working `uvx`/`pipx` one-liner. This is *decision effort, not build effort* — `polylogue-3tl.7` already frames it as "release is only the decision to merge." Do this first; it unblocks everything below.

2. **Install-matrix proof + doc reconciliation (`3tl.7`, `3tl.13`).** After the tag: a scheduled fresh-environment smoke (uvx / pipx / brew / `docker run` / `nix run` → `demo seed` → one `find` → one `read` → `--version`) so the *first stranger is not the test*, and a same-PR sweep replacing every "not available until released" hedge in `installation.md`/`getting-started.md`/README with the real one-liner. State the Windows story honestly (WSL2-only) and drop macOS if untested — an honest narrow matrix beats a broad broken promise. Highest-ROI *engineering* signal: "install just worked, on a cold machine" is the single strongest competence proof a stranger gets.

3. **The de-meta README pass with reproducible claims (`3tl.12`).** The operator flagged this directly (raw-log 2233): meta sentences like "Memory-benefit claims stay capability-phrased until measured uplift experiments prove the effect" do not belong in a README. Strip all meta/persuasion; replace agent-coined jargon ("judged notes," "work phases" — he suspects the latter is insight-slop) with plain nouns or delete; and for every retained claim, ensure it's demonstrable via `polylogue demo tour` or a cited finding. This is pure editing, high signal: a README that over-explains its own honesty reads as *insecure*; one that just shows the thing reads as *competent*.

4. **One flagship live-cited finding, regenerated on the current archive (`3tl.11`).** The agent-forensics finding is stale (v23, retired); regenerate on v24 so cited counts match `polylogue` live reads. One artifact where a stranger clicks a URL and sees a real, non-trivial claim backed by queryable evidence *is* the "cite a published finding" leg of the epic's terminal state. This is the proof that the substrate does something ordinary transcript folders cannot.

5. **The screencast (`3tl.5` recordings already exist — publish/place them).** A 60–90s query-tour tape embedded at the top of the README. Moving pictures collapse the "what even is this" gap faster than any prose. It's already recorded; the marginal effort is placement, not production.

**DO NOT BUILD (say no explicitly):**

- **A separate docs site as a launch gate.** Pages workflow exists; a polished microsite is `P3` (`3tl.4`/`3tl.8`) and its marginal effect on a *stranger's competence judgment* is low relative to a working install + one real finding. GitHub repo page + README carries the launch. Ship the site later or never.
- **The Datasette exhibit, the versioned interchange schema, the multi-model leaderboard, the codebase atlas** (`polylogue-45i`, `3tl.6`, `3tl.3`, `3tl.y0b`). All genuinely cool, all `P3`, all zero-to-negative marginal signal for a first impression, all effort sinks. They are "impressive to people already convinced." Backlog.
- **Any new capability for launch.** The wedge (MCP-recall continuity, two-agent-same-repo) is already built. Launch surfaces what exists; it does not motivate a feature sprint. Resist "one more thing first."
- **A memory/uplift *marketing* claim.** See §3 — the experiment currently says packs lose. Do not build a narrative around measured memory uplift until `cfk` re-runs positive.

---

## 3. The staged launch sequence (a staircase, each step gated on the last)

Four steps, strictly ordered. Do not compress; each de-risks the next.

**Step 1 — Release cutover.** Merge the release-please PR; first tag fires; PyPI + Homebrew + GHCR + FlakeHub publish; install-matrix goes green **two consecutive weekly runs** (the `3tl.7` acceptance gate) *before* anyone is invited. Reconcile install docs in the same window. **Exit gate:** a stranger can `uvx polylogue demo tour` and get a result in 30s. Until this is true, there is nothing to launch — a broken install on launch day is the one unrecoverable failure.

**Step 2 — uvx one-liner + doc reconciliation, quietly live.** No announcement yet. Let the repo sit correct for a few days: install works, README de-meta'd, one finding cited. This is the "cold-reader pass" the epic's terminal state demands (`3tl` AC: verified by someone/something with no context). Run an actual cold reader — a fresh agent or a person — against the README-first-screen → demo → cited-finding path. Fix what they trip on. **Exit gate:** cold reader completes all three legs unaided.

**Step 3 — the data-story essay, live-cited.** One essay built on `3tl.11`'s regenerated finding: "here is what I found in my own N-thousand-session archive that a transcript folder can't tell you" — costs across providers, agent claim-vs-evidence gaps, resume/abandonment patterns — every number a live-queryable citation. **Honesty constraint (load-bearing):** the essay's spine is *forensics on real data*, NOT *memory uplift*, because `cfk` currently records "packs lose." Memory-continuity is described as a **capability** ("agents can query their own history via MCP"), not a **proven benefit**, until `cfk` re-runs n≥3 positive. If `cfk` lands positive before the essay, fold in the uplift result *with its data* — that becomes the strongest possible version and, per the epic, "the natural launch post." If it doesn't, ship the forensics essay anyway; it stands alone.

**Step 4 — one sharp thread into the AI-memory discourse.** A single, non-spammy thread (the operator already noted a candidate hook, raw-log 2243) pointing at the essay, leading with the *differentiated wedge*: **agents reading their own structured session history through MCP — memory as a queryable substrate, not a vector-dump.** The two-agent-same-repo (s7ae) demo is the jaw-drop attachment. One thread, one essay link, one demo. Not a campaign. The narrative is "solo-built, local-first, rigorously-modelled, here's the proof" — which travels precisely *because* it isn't marketing.

Ordering rationale: install-before-invite (never make the first user the test), quiet-before-loud (cold-reader before broadcast), findings-before-thread (the thread points at a proof, not a promise), and memory-claim gated on `cfk` throughout.

---

## 4. First-10-users profile + the discourse communities

**First-10-users profile — who actually installs a local Python archive for AI sessions:**

- **The operator himself, counted honestly as user #1.** The success metric is his own daily use (raw-log 2129/2134). If the dogfood loop isn't eager, the other nine don't matter.
- **Heavy coding-agent operators** running Claude Code / Codex / multiple agents daily, who already feel the pain of history vanishing into vendor silos and manually shuttling context between GPT-Pro, coding agents, and notes (raw-log 2134/2195). They have the exports, the CLI comfort, the MCP setup, and the felt need. This is the entire addressable first cohort.
- **Local-first / self-hoster / "own your data" practitioners** — people who run their own everything and are allergic to SaaS memory products. Local-first *is* the pitch to them.
- Secondarily: **agent-tooling builders and researchers** studying agent behavior, who want a structured corpus of real agent runs (the forensics finding is bait for exactly this group).

Not in the first 10, don't chase: casual ChatGPT users, teams, anyone who wants a hosted dashboard.

**The 2–3 discourse communities the narrative travels in:**

1. **The AI-agent-memory / context-engineering discourse** (X/Twitter + the practitioner blogosphere). This is where the wedge lands: everyone is shipping vector-RAG "memory," and Polylogue's counter-position — *structured, queryable, evidence-linked session history that agents read via MCP* — is a sharp, differentiated take. The essay + one thread live here.
2. **Local-first / self-hosted-AI communities** (r/LocalLLaMA, Hacker News "Show HN," self-hosting forums). Local-first + no-API-key + reads-files-you-already-have is a native fit; "git for the work around your code" is the HN-legible framing.
3. **The Claude Code / Codex power-user community** (Discords, the coding-agent tooling niche). Smallest but highest-conversion: they have the exact exports and the exact pain. The two-agent-same-repo demo is aimed straight at them.

Travel order matches §3: seed in (3) where conversion is highest and feedback is best, let it carry to (1) via the essay, and only then consider a Show HN in (2) once the install is bulletproof (HN punishes broken installs mercilessly — which is why Step 1 gates everything).

---

## 5. Monetization stance: defer, explicitly, so no one builds a funnel.

**Stated position: no monetization surface is built or planned. Polylogue is free, MIT, local-first, with no server to charge for. Do not build signup, billing, telemetry, tiers, or a "pro" anything.**

The instrumental value is **not revenue** — it is (per the prior-wave findings and raw-log): a **fundable capability proof**, **resume-gap legitimacy**, and **genuine personal utility**. Those are served by the project *existing and being legible*, not by a payment funnel. Concretely:

- **Leverage vector = employment / grant / patronage, not sales.** A rigorously-engineered, publicly-inspectable, solo-built system-of-record with published findings is a portfolio artifact that speaks to hiring managers and grant reviewers directly. That audience judges the *code and the findings*, and a monetization funnel actively *weakens* that signal (it reads as "trying to sell a side-project" rather than "built serious infrastructure").
- **If any money path is ever entertained, it is patronage/sponsorship** (GitHub Sponsors button, zero obligation) or grant application — both of which are *pull* mechanisms that require no product change and no funnel. Not a subscription, not hosting, not a team tier.
- **The anti-instruction for anyone downstream:** do not build signup flows, waitlists, usage analytics, or conversion funnels. They contradict the local-first invariant (there is no server to measure from), they distort the identity decision in §1, and they are premature by definition — the project has zero external users. Ship the tool, publish the findings, keep it free.

---

**One-line summary for the operator:** the stack is already built and idle — the entire remaining game is *fire the tag, prove the install, strip the meta, publish one real finding, tell one honest story* — with every memory claim gated on `cfk` re-running positive, and nothing that smells like a product.

---

## [af766bbf57d21dca7] Red-team: the case against Polylogue as a category and a project

# Red-team: the case against Polylogue as a category and a project

- **"System of record for AI work" is a category few operators actually want** — Records-of-record earn their keep only when something downstream *depends* on them (audit, billing, compliance, legal discovery). For a solo dev, AI transcripts are exhaust, not assets; nobody gets fired for losing last month's Claude Code sessions. The project assumes a demand that its own single-operator dogfooding can't validate. — Concede partially: the honest scope is "personal continuity + cost accounting for one power user," which is real but ~2 orders of magnitude smaller than "system of record." The vision language oversells the demonstrated need.

- **The provider is the natural system of record, and it's winning** — Anthropic, OpenAI, Google already retain, search, and (increasingly) let you resume/branch your own sessions server-side. Every quarter they ship native history, projects, memory. A local mirror is racing incumbents who own the write path and the identity. — Rebuttal: cross-provider unification is the one thing no incumbent will ever build (they're structurally adversarial to it), and local-offline survives account bans/deletions/API deprecations. But that only matters to the tiny slice who genuinely work across 4+ providers *and* distrust vendor retention.

- **Idempotent-content-hash + 5-tier split SQLite is a database engine masquerading as an archive** — Generated-column identity, contentless FTS5 with three sync triggers, lineage tail-storage with recompose-on-read, blob GC with dual safety invariants, two schema-evolution regimes. This is more storage engineering than most funded startups carry, in service of a read-mostly personal log. — Why it bites: every one of these is a maintenance liability with a bus factor of one. The lineage recompose-on-read alone (branch_point deliberately-not-a-FK, cascade-ordering hazard) is a footgun documented *because it already fired*. — Concede: the dedup problem (#2467, 16K physical vs 8.8K logical) is real and provider-caused; but the answer might be "store the dup and don't care" — disk is cheaper than the invariant.

- **The lineage-normalization jewel optimizes storage nobody is paying for** — Storing only the divergent tail to avoid replaying parent prefixes saves bytes on a 38GB archive. Bytes are free. The complexity tax (quarantined cycle-breaks, resolve-on-save topology edges, "deliberately not a FK") is paid in cognition and bug surface forever. — Rebuttal: it's not (only) storage — correct authored-user/cost accounting *requires* not double-counting replayed prefixes. — Counter: you can dedup for *counting* without physically normalizing *storage*; the counting fix is a GROUP BY, not a schema regime.

- **Queries-as-objects / result-set-algebra is a query engine the project explicitly said it wouldn't build** — `query-pipeline-substrate.md` opens with "without becoming its own query engine," then specifies a typed AST, lowering layer, traversal stages, recursive CTEs, aggregation stages, and a completion metadata registry. That *is* a query engine. #2544 "result set algebra" is the tell. — Why it bites: a Lark DSL with 10 predicate node types and 7 pipeline stages, for a corpus one SQLite `LIKE` + a dozen saved SQL views would serve, is textbook YAGNI. — Concede narrowly: FTS + `origin:` + `since:` field syntax is genuinely useful; everything past `find … then <action>` is speculative surface built ahead of demonstrated queries.

- **The "just grep it" minimalist has a near-unanswerable objection** — `rg` over `~/.claude/projects/*.jsonl` answers "what did I do / what did the agent say" in 50ms with zero daemon, zero schema, zero convergence debt. Add `jq` for structure. That's 80% of retrieval value at 0.5% of the code. — Rebuttal: grep can't do cross-provider, can't cost-account, can't dedup lineage, can't semantic-search. — Counter: it can do the *retrieval* people actually perform daily; the other capabilities are the ones whose demand is unproven. The minimalist wins on P50 use, loses on P99 use, and the backlog is 90% built for P99.

- **The hot daemon is standing infrastructure for a batch problem** — `polylogued` runs continuously (convergence stages, hot-file quiet deferral, ProcessPoolExecutor, embedding catch-up, cursor-lag telemetry) to ingest files that a cron job or a `SessionEnd` hook could sweep on demand. — Why it bites: a resident process is a resident liability — memory, crashes, restart semantics, the #1498 cascade retro, WAL band-aids. — Rebuttal: browser-capture and live tailing genuinely need a resident receiver. — Concede: the *capture* channel justifies a listener; the *convergence* machinery doesn't need to be daemon-resident and could run post-hoc, halving the surface.

- **Recursive/insight materialization builds analytics for an N=1 user who will read them once** — Session profiles, timelines, threads, summaries, topology, pathologies, phases, latency profiles, workflow-shape distributions — six insight readers + a pathology miner + postmortem bundles. For one operator, most of these get run in a demo, screenshotted, and never queried again. — Why it bites: each insight is a stored derived model with its own schema, convergence stage, CLI verb, MCP tool, and JSON schema. The materialization cost is permanent; the read frequency is ~once. — Concede: `session_costs`/`provider_usage` earn their place (recurring operator question). The behavioral-analytics tier is a museum.

- **~130 MCP tools is a surface no agent can navigate and no human can maintain** — Tool-choice degrades with tool count; 130 tools across `server_*.py` means the agent-facing "continuity surface" is itself a discovery problem. Every add requires `EXPECTED_TOOL_NAMES` + a contract + schema regen. — Why it bites: this is the opposite of legible — it's sprawl with lint scaffolding to manage the sprawl. — Rebuttal: agents subset via tool-search. — Counter: then most of the 130 are dead weight in practice, and the maintenance:usage ratio is damning.

- **The privacy hawk's objection: a single unencrypted local archive of all your AI work is a honeypot** — `user.db` is "durable, irreplaceable"; the whole set is plaintext SQLite. One laptop theft, one backup misconfig, one malicious MCP-connected agent with `--role write`, and your entire cross-provider intellectual history — including everything you told an AI you'd never say aloud — is exfiltrated in one file copy. The providers at least shard risk across their security teams. — Why it bites: centralization *is* the value proposition and *is* the attack surface; they're the same design choice. — Concede: no at-rest encryption story surfaced in the architecture; "local = private" conflates local with secure. This is a real gap, not a rebuttal-able one.

- **Provider→origin retirement being "in progress, not done" is a symptom, not an incident** — A core vocabulary split (Provider vs Origin vs Source, non-injective GEMINI+DRIVE→AISTUDIO_DRIVE, a `project_origin_payload` shim, tracked across 4 beads) has been mid-migration long enough to become load-bearing. — Why it bites: it signals the project accretes concepts faster than it retires them — the exact failure mode that kills solo-maintained systems. The backlog grows; nothing closes. — Rebuttal: it's carefully tracked debt. — Counter: tracked debt is still debt, and "carefully tracked" is how a one-person project rationalizes never paying it.

- **"Cross-provider" is a moving target that guarantees permanent parser churn** — Seven providers (Claude web+Code, ChatGPT, Codex, Gemini/Drive, Antigravity, Hermes, Grok reserved), each with an undocumented, unversioned, unilaterally-changing export shape. The 139-retry single-doc parser bug and the shape-based tightness-ordered detector are evidence the parsing layer is inherently unstable. — Why it bites: the maintainer is on a treadmill of other companies' format changes, forever, for formats they don't control. — Concede: this is structural to the category — anyone doing cross-provider unification signs up for it. It's a reason the category is thankless, not a project-specific flaw.

- **"Recursive safety" and adversarial-closure ceremony are governance overhead for a repo with one committer** — AC-matrices, adversarial-loop skills, assured-close, proof packs, coherence-gap reviews, 15+ `docs/plans/*.yaml` lint manifests each guarded by a devtools verify lane. — Why it bites: process weight calibrated for a distrustful multi-team org, imposed on a solo project, is pure friction — it slows the one person who already has full context. — Rebuttal: it's what lets a solo dev safely run 16-agent swarms without regressions. — Concede partially: the swarm workflow *is* the justification; but then the process serves the *meta*-project (agent orchestration research) more than Polylogue's users, which is worth being honest about.

- **The measure-algebra / honest-cost accounting is precise about a number that's directionally wrong anyway** — Enormous care goes into `material_origin` gating, disjoint token lanes, credit-vs-API pricing, LiteLLM catalog matching. But `cost_usd` is "API-list-equivalent" and *overstates subscription spend* by construction; cache reads are free on Max/Pro; the real question ("am I getting value from my $200/mo") isn't a token sum at all. — Why it bites: the algebra is rigorous about inputs to a metric that doesn't answer the operator's actual question. — Rebuttal: honest per-lane accounting is prerequisite to *any* correct view. — Counter: prerequisite ≠ sufficient, and the marginal rigor (7.69× bug notwithstanding) is chasing decimals on a fundamentally advisory figure.

- **The whole edifice is a solution whose best customer is its own author's agent-research agenda** — The strongest actual use ("continuity = MCP surface not CLI; assertions > CLAUDE.md," per session memory) reveals Polylogue's real job is *feeding context to future Claude/Codex agents* — i.e., it's infrastructure for the operator's other project (running agent swarms), not a standalone product. — Why it bites: as a standalone "system of record," it has an addressable market of roughly one; as agent-memory-substrate it's real but should be scoped and named as *that*, killing most of the archive/analytics/query-engine backlog. — Concede: this is the strongest reframe — 80% of the value is "durable, queryable agent memory + cost visibility," and ~70% of the backlog (insight museum, query algebra, lineage normalization, portfolio/export cluster) is orthogonal to it.

- **Simplest thing delivering 80% of the value** — (a) a `SessionEnd` hook that copies each provider's JSONL into one flat dir; (b) FTS over `search_text` with `origin:`/`since:` filters; (c) three saved SQL views: cost-by-day, sessions-by-repo, resume-candidates; (d) one MCP tool `search(query)` returning message spans. No daemon-resident convergence, no lineage storage normalization, no query DSL, no insight materialization tier, no 130 tools. — Why it bites: this fits in ~2K lines, has bus-factor-survivable complexity, and makes the split-tier schema, the algebra, the composer, and most beads pointless. — Concede: it loses semantic search, browser-capture, honest lineage counting, and cross-provider dedup — the four things worth *incrementally* adding back, each only when a live query demands it.

---

**GPT-pro adversarial prompt stubs:**

- **[A]** "You are a skeptical staff engineer reviewing a solo-maintained 'local cross-provider system of record for AI work' built on a 5-tier split-SQLite schema with lineage-tail-storage, a Lark query DSL lowering to recursive CTEs, ~130 MCP tools, and a resident convergence daemon. Argue the strongest case that this is over-engineered relative to `rg + jq + 3 saved SQL views`, and identify the single subsystem whose removal would most reduce maintenance risk while preserving >70% of realized operator value. Steelman the maintainer's counter, then rule on it."

- **[DR]** "Research whether 'personal cross-provider AI-transcript archive' is a durable product category or a transient gap. Track how fast Anthropic/OpenAI/Google are shipping native history, projects, memory, resume/branch, and cross-session recall (2024→2026), and how their ToS treat third-party export. Assess: what fraction of a local archiver's differentiating features (unification, resume, cost accounting, semantic search) will incumbents commoditize within 18 months, and what — if anything — is structurally impossible for a provider to build? Conclude whether the defensible residual justifies the engineering surface."

- **[A]** "A one-person project's real use of its own 'AI system of record' turns out to be feeding durable context/memory to the author's *other* work (running large agent swarms), not standalone recall. Assume that's true. Redesign the scope adversarially: what's the minimal 'agent-memory + cost-visibility substrate' that serves that job, and which existing subsystems (insight materialization, result-set algebra, lineage normalization, export/portfolio cluster) does it render dead weight? Name what breaks if each is deleted, and whether that break is ever actually hit."

---

## [acd90cd1749397d3e] Ideas

Grounded: read the insight registry (10 registered types — profiles carry `engaged_minutes`/`tool_active_minutes`/`tool_calls_per_minute`/`thinking_duration_ms`/`tool_duration_ms`/`workflow_shape`/`terminal_state`; work_events carry labeled start/end/duration spans), the `polylogue-9l5` tower map (measure-registry composition rule: every layer = measure × grouping × window × comparison × uncertainty, never bespoke analyze modes), and the operator's raw-log. Key operator asks driving this zone: **1841** ("wallclock-time blindness… 1min editing, 15min compilation, 45m full test run… whether work is inherently that way or agents guess ridiculously inefficiently"), **2134** ("mass-grab every sinex chatlog in compact view, feed to gptpro"; "analysis of suboptimal ways agents do stuff, incl. temporal"), **2001** (skillful-grep vs blind-whole-file reads), **1944** (local-LLM analysis — gemma 10GB / deepseek / flash-lite), **1804** (wallclock + token analysis).

## Ideas

- **Gap-taxonomy materializer** — decompose each session's inter-message wallclock into typed spans (edit / compile / test / rebuild / model-thinking / human-idle) by joining `tool_result` end-timestamps to next-event start + tool identity + command regex; emit a `session_gap_profile` insight with a span-ledger. Directly answers 1841's literal ask; substrate (per-tool `duration_ms`, timestamps) already exists — pure new materializer over blocks. — polylogue-9l5.8 (temporal)
- **Long-call time-sink ranking** — surface the specific tool calls whose wall-duration exceeds N min, labeled by command class (full-test / nix-switch / rebuild) from the `tool_use` input, ranked per session and rolled up per repo — turns "hours of blindness" into "45m × 3 wasted full-suite runs on file X". `session_tool_timing` exists; add duration-bucketed command labeling. — NEW
- **Retry-storm / guess-and-check detector** — sequence pathology: consecutive edit→same-target-test→edit cycles where `tool_result_is_error` (v16 keystone) stays true and the error signature doesn't change = churn; materialize as `retry_storm` pathology with cycle count + wasted minutes. Process-mining-lite over the `actions` view. — polylogue-9l5.9 / epidemiology
- **Agent-efficiency score with peer baseline** — normalize `tool_active_minutes / wallclock` and productive-edits-per-hour against a per-repo, per-`workflow_shape` baseline so a slow session is judged against *like* work, not globally. Both numerators live on `session_profiles`; the non-obvious bit is the cohort-relative z-score as a registered measure. — polylogue-9l5.2 / .7
- **Tool-error recovery latency** — from first error `tool_result` to next success on the same tool/target: turns-to-recover and wallclock-to-recover, as a distribution per tool and per model. Distinguishes "recovers cleanly" from "spirals". Uses keystone; one measure over the action view. — NEW
- **Context-gluttony index (blind-read vs targeted-grep)** — ratio of whole-file `Read` calls (no offset/limit) and bytes ingested to `Grep`/`Glob` targeted queries; flags 2001's "reads huge files instead of skillful grepping" pattern per session/agent. Pure `tool_use`-input measure. — NEW
- **Cost-per-outcome join** — the sharpest "so what": join `terminal_state` (success/abandoned/stuck) to `session_costs` → "USD per *successful* session" and "spend burned on abandoned+stuck sessions" per model/repo/week. Both surfaces exist; needs only the outcome-conditioned join as a measure. — polylogue-9l5.1
- **Delegation yield (subagent ROI)** — over topology Task-spawn edges: tokens/cost/wallclock consumed inside child subagents vs whether the *parent* reached terminal success; "was the delegation worth it". Joins child cost to parent outcome across `session_links`. — NEW (delegation)
- **Rework half-life / churn survival** — LAYER-4 survival curve: time until a file touched in session X is re-edited in a later session; short half-life = code that never stabilizes, a leading indicator of thrash. Cross-session file-touch extracted from tool inputs. — polylogue-9l5.9 (survival)
- **Corpus-compaction pack for external LLM** — `find <query> | compact` projection emitting a token-budgeted, decision-dense digest (drop tool-spam blocks via `material_origin`, keep human turns + decisions + errors + outcomes), sized to a target context window with a manifest of what was dropped. Directly ships 2134's "feed gptpro" workflow; projection over existing read-view + origin filtering. — NEW
- **Measure→prose narration (local-LLM)** — `analyze --narrate` runs a registered measure, passes only the compact numeric result to gemma/deepseek to produce grounded prose ("70% of wallclock in test runs; retried the failing test 4×; recovered on attempt 5"). LLM narrates measures, never invents facts — enforces the composition rule as a guardrail. — polylogue-9l5 semantic (mhx.5) / 1944
- **Pathology epidemiology over time** — prevalence of each pathology (retry-storm, gluttony, abandonment) per ISO-week and model, with changepoint detection flagging when a model upgrade shifts the rate. `get_pathologies` exists; add the temporal rollup + changepoint as measures. — polylogue-9l5.8 + epidemiology
- **Shape-constrained failure cohort retrieval** — "find sessions structurally like *this* stuck one": combine `workflow_shape` + embedding neighbors + `terminal_state=stuck` to return a cohort ready to feed an LLM for cross-session root-cause. `find_similar_sessions` + embeddings exist; new bit is the shape+outcome-constrained retrieval measure. — NEW
- **Context-switch / interruption tax** — from `session_links` + timestamps: detect abandon-A → spawn-B → return-to-A patterns, count switches per day, estimate re-orientation cost (resume→first-productive-edit latency). Mirrors operator's own task-initiation friction (raw-log 1497). — NEW
- **Thinking-vs-doing drift** — `thinking_duration_ms` vs `tool_duration_ms` trended per model over time; flags a model that starts over-reasoning (reasoning tokens climb, edits flat) — an early "this model got worse for my work" signal. Both fields exist on profiles/timeline. — polylogue-9l5.8
- **Abandonment autopsy clustering** — cluster abandoned/stuck sessions by (last-tool-before-abandon, last-error-signature) to surface the top failure modes that make the operator quit. `find_abandoned_sessions`/`find_stuck_sessions` exist; add the terminal-signature cluster measure. — polylogue-9l5.1
- **Wasted-test-run detector** — cross-ref gap-taxonomy long test spans with whether an error→fix actually followed: a 45m full-suite run on a change that produced no caught error = wasted minutes (the CLAUDE.md blanket-run anti-pattern, quantified). Composes on the gap-taxonomy materializer. — NEW
- **Per-session "so what" scorecard panel** — one composed `reader_panel` stacking efficiency-score + gap-taxonomy sparkline + retry-storm count + cost-per-outcome + delegation-yield, all sourced from the registered measures above — the single-session dashboard the operator keeps missing. `reader_panel` slot already on `InsightType`. — NEW

## Strongest 3 — GPT-pro prompt stubs

**[A] Gap-taxonomy temporal decomposition materializer.** "In Polylogue (SQLite `index.db`, three-level sessions→messages→blocks, `blocks` carry per-tool `tool_duration_ms` and `tool_result_is_error`; profiles already expose `tool_active_minutes`/`wall_duration_ms`), design a `session_gap_profile` materializer that partitions each session's wallclock into typed spans (edit / compile / test / rebuild / model-thinking / human-idle) from inter-event timestamp gaps + tool identity + command-string classification. Specify: the gap-classification decision table, how to distinguish human-idle from long-running tool spans, the STRICT-table schema for the span ledger, and how it registers as a measure (grouping × window × uncertainty) with construct-validity metadata (evidence tier, coverage precondition when timestamps are sparse). Output the ledger, per-class minute totals, and a compact sparkline string."

**[DR] Outcome-conditioned agent-efficiency & pathology framework.** "Survey the state of the art in quantifying AI-coding-agent efficiency and failure modes from session transcripts: retry-storms/guess-and-check, tool-error recovery latency, context-gluttony (whole-file reads vs targeted grep), cost-per-successful-outcome, delegation ROI, and abandonment autopsy. For each, give the operational definition, the minimal event signal needed, known confounds, and how to express uncertainty honestly (small-sample, sparse-timestamp). Recommend which are robust as automated pathology detectors vs advisory-only, and how to trend their prevalence over time with changepoint detection on model upgrades."

**[A] Corpus-compaction pack for external-LLM handoff.** "Design a `find <query> | compact` projection for Polylogue that turns a cohort of AI sessions into a single token-budgeted, decision-dense digest for feeding GPT-pro/Gemini: drop tool-spam blocks via `material_origin`, preserve human-authored turns + explicit decisions + error→fix pairs + terminal outcomes, dedup replayed lineage prefixes, and fit a target context window. Specify the block-selection scoring, the drop-manifest, per-session boundary markers, and how the packer reports what fidelity was sacrificed to hit the budget."

---

## [a1fff6fe4127bed68] Polylogue security/privacy threat model — red-team enumeration

Grounded in `daemon/http.py`, `browser_capture/{server,receiver}.py`, `docs/daemon-threat-model.md`, and the `polylogue-kwsb` security epic (+ `jnj.5` reset bug). Findings:

**Polylogue security/privacy threat model — red-team enumeration**

- **DNS-rebinding read of the whole archive** — [REAL, exploitable] HIGH. GET routes in `_dispatch_get` enforce *no* Origin/Host check (only POST calls `_check_cross_origin`); auth defaults OFF on loopback and `_client_host` is checked, never the `Host` header. A page on evil.com that rebinds its hostname to `127.0.0.1` becomes same-origin to the daemon in the browser's eyes and can `fetch` `/api/sessions/:id/read` and read every response. — Mitigation: require a `Host: 127.0.0.1|localhost` allowlist on *all* requests; make the loopback bind non-guessable-port + token by default. New kwsb child.

- **Malicious local process POSTs forged captures into the spool** — [REAL, exploitable] HIGH. `_origin_allowed` returns `True` when `Origin` is `None` (any non-browser client), and the receiver has no auth by default. `curl -XPOST /v1/browser-captures` with a valid envelope shape writes a fabricated session artifact that the daemon then ingests → archive/context poisoning of otherwise-trusted history. — Mitigation: bind receiver to a token even on loopback; treat spool artifacts as untrusted-until-provenance-verified. kwsb.

- **Prompt-injection in an archived transcript → MCP-write self-laundering** — [REAL, design-level] HIGH. The MCP server runs with the `write` role; an agent reading an adversarial transcript (`get_messages`/`search`) can be steered to call `add_tag`/`record_correction`/`set_metadata`/`delete_session`, writing attacker-chosen state into the irreplaceable `user.db` — laundering injected claims into "operator" assertions. — Mitigation: the `37t.11` trust-class doctrine (SYSTEM/QUOTED/OPERATOR, no verbatim injected text as directives, judgment-gate for OPERATOR promotion) — must ship *before* injection legs. Red-team fixture in kwsb.

- **Exfil via a compromised/injected read-role agent** — [REAL] HIGH. The MCP surface (~130 tools) hands an agent the entire cross-provider history; a single injected instruction plus any network-egress tool exfiltrates it. Read access is total by construction. — Mitigation: egress-free MCP sandbox; per-session scoping; audit log of bulk `search`/`get` calls. Partly inherent.

- **Pasted secrets (API keys/tokens) re-surface, FTS-searchable, plaintext** — [REAL] HIGH. Threat-model doc line 84 admits secrets are "stored as-is"; they land in `blocks.search_text` and the contentless FTS index, so any read surface (CLI `find`, MCP `search`, web shell) resurrects them. — Mitigation: excision + secret-redaction, the core of `kwsb` ("forget on purpose, never leak secrets"); detector at ingest.

- **Plaintext archive as a high-value on-disk target** — [REAL] HIGH. Five `.db` tiers are unencrypted SQLite; the doc lists encryption-at-rest as an explicit non-threat (delegated to OS disk crypto). A backup leak, cloud-sync, or stolen disk yields the full corpus of every AI session. — Mitigation: document the disk-encryption dependency loudly; consider optional at-rest encryption for `user.db`/`source.db`. Residual/accepted.

- **`ops reset --session/--source` tombstones before the `--yes`/preview gate** — [REAL bug, tracked] MED-HIGH. `jnj.5`: a typo'd ref mutates irreplaceable `user.db` suppression state with no dry-run and no confirmation. — Mitigation: route identity resets through the shared mutation contract (dry-run → `--yes` → stable JSON), per kwsb AC. Fix is scoped.

- **Web-shell bootstrap served unauthenticated on loopback** — [REAL, needs verify] MED. `_check_shell_bootstrap_access` serves `/`, `/s/:id`, `/p`, `/a` HTML with no token when host+client are loopback; the page's JS then calls the "authenticated" API — so either the token is embedded in unauthenticated HTML (token leak to any loopback fetch) or the shell silently downgrades to the open API. — Mitigation: confirm the shell never embeds the bearer token in bootstrap HTML; gate shell HTML behind the same token when one is set.

- **Non-constant-time bearer-token comparison** — [THEORETICAL] LOW. `_check_auth_logic` uses `auth_header[7:] != auth_token` (and receiver `auth[7:] == config.auth_token`) — a plaintext, non-`hmac.compare_digest` compare, a timing oracle in principle. Loopback-only, low value. — Mitigation: `hmac.compare_digest`. Cheap hardening.

- **Auth token accepted as `?access_token=` query param** — [REAL, minor] LOW-MED. `_check_auth` falls back to a URL query token for EventSource; URLs leak into referrers, shell history, and any proxy/log despite `log_message` being suppressed. — Mitigation: restrict the query-token fallback to the SSE `/api/events` route only; short-TTL scoped tokens.

- **Spool disk-fill / capture-flood DoS** — [REAL, local] MED. `MAX_BROWSER_CAPTURE_BODY_BYTES = 128 MiB` per POST with no rate limit or spool quota; a local process loops POSTs and exhausts disk, degrading the daemon and the host. — Mitigation: per-origin rate limit + spool size cap + backpressure. New bead.

- **Outbound post-command injection when posting is enabled** — [REAL when flag on, else GATED] MED. `poll_post_commands` dispatches any `*.json` manually dropped in the queue dir; with `POLYLOGUE_BROWSER_POST_ENABLED=1` a local writer can drive the extension to type/send messages into live ChatGPT/Claude threads (`submit=true`). Default-OFF and `enqueue`/`poll` both re-check the flag, which contains it. — Mitigation: keep default-OFF; sign/authenticate queue entries; per-command operator confirmation. kwsb-adjacent.

- **Unauthenticated `/metrics`, `/healthz/*`, OTLP receiver** — [REAL, minor] LOW. Metrics/health are auth-free by design (scraper convention) and derive counts from the index DB; the OTLP `/v1/*` receiver (when `observability_enabled`) accepts unauthenticated loopback POSTs, letting a local process pollute `ops.db` otlp/metrics. No content leak, but count-level inference + telemetry poisoning. — Mitigation: optional token even for scrape; validate OTLP source. Accept/minor.

- **Supply chain via prebuilt wheels** — [REAL, environment-dependent] MED. CLAUDE.md/threat-model lean on "pre-built wheels" and the Nix closure for integrity; outside NixOS (pip install) a compromised `orjson`/`vec0`/pydantic wheel executes in the sole-writer daemon process with full archive + browser-post authority. — Mitigation: hash-pinned lockfile + `--require-hashes`; document the Nix-closure assumption as a hard dependency, not a nicety.

- **Multi-instance / concurrent spool + writer races** — [MOSTLY THEORETICAL] LOW. Two receivers or a receiver + manual writer share the spool; `write_capture_envelope` uses `tempfile → os.replace` (atomic) and content-hash-deterministic paths, so corruption is unlikely, but a second unmanaged writer to the archive DBs violates the single-writer invariant the whole design rests on. — Mitigation: single-writer lockfile/PID guard on the daemon + spool. Low priority.

- **Ingest route as arbitrary-file-copier — CLOSED as safe** — [MITIGATED, noted for completeness] `_staged_inbox_source` resolves only names already present in the inbox and enforces `resolved.relative_to(inbox_root)`, blocking `../` traversal and arbitrary local-file ingestion via the loopback API. Good existing control; keep the test.

---

**GPT-pro prompt stubs**

- **[A]** "Given this Python `BaseHTTPRequestHandler` daemon that binds `127.0.0.1` with an optional bearer token and Origin-allowlists only POSTs, write the exact DNS-rebinding attack page + a `Host`/`Origin` allowlist middleware that closes it on GET and POST without breaking the same-origin loopback web shell. Include a test that a rebound `Host: evil.com` request is rejected."

- **[A]** "Design a two-tier trust model for injecting recalled AI-transcript content into an agent's context (SYSTEM structural facts vs QUOTED attributed data vs OPERATOR directives), such that no verbatim tool-output/web text can ever be emitted as a directive and a source without a judgment gate is type-incapable of emitting OPERATOR items. Give the Python protocol types and a property test that assembled output never contains unfenced QUOTED content."

- **[DR]** "Survey the state of the art in threat-modeling *local-first, single-user* applications that expose loopback HTTP APIs, MCP/agent tool surfaces, and browser-extension capture channels (2024–2026). Cover: why loopback is not a trust boundary (same-host processes, DNS rebinding, malicious extensions), secret-material handling in personal data lakes, prompt-injection→tool-abuse laundering through agent memory stores, and encryption-at-rest tradeoffs. Produce a prioritized control catalog with citations and note which controls are inherent-residual for single-user tools."

---

## [a5e61d6bfbf4a6e6a] DOCS & ONBOARDING AS A PRODUCT — 14 ideas grounded in the live docs surface (`docs/README.

DOCS & ONBOARDING AS A PRODUCT — 14 ideas grounded in the live docs surface (`docs/README.md` map is generated; `verify_doc_commands.py`, `demo tour`, `search.md` DSL, `glossary.md` entry-layers, `query_explain.py`/`parser_diagnostics.py` already exist).

- **Zero→first-value "golden path" doc that runs on the demo corpus, not on private data** — `getting-started.md` still opens with `git clone … && nix develop`, gating first value behind a Nix toolchain and a real archive; a stranger can't "run+cite" until they've imported their own history. Ship a `docs/first-value.md` whose every command is `polylogue demo …` so the cold reader reaches a real search result + proof artifact in one copy-paste block. — NEW
- **Make `verify_doc_commands.py` execute the golden-path blocks, not just resolve tokens** — the lint proves a command *name* exists but never that the documented invocation *runs*; a flag rename or output-shape change sails through. Add a doc-tested lane that actually executes fenced blocks tagged `example` against a throwaday demo archive and diffs stdout, so docs fail when the tool changes. — NEW
- **A task-oriented "How do I…?" layer indexing the reference** — docs are organized by *subsystem* (architecture/internals/daemon), but a cold user thinks in tasks ("find what an agent broke last Tuesday", "see what a session cost", "resume abandoned work"). Author `docs/howto/` recipes, each: question → one query → expected shape → deeper-reference link. This is the missing middle between README's five questions and the generated CLI reference. — NEW
- **Teach the query DSL by worked example, promoted out of the grammar dump** — `search.md` leads with EBNF (`compact-query ::= …`); the Lark grammar is honest but is a spec, not a lesson. Add a top-of-page "10 queries that teach the grammar" ladder (bare term → field clause → `since:7d` → `sessions where … | group by … | count`) each annotated with what it *selects*, before the formal grammar. — bead polylogue-jnj.7 (help leakage) adjacent
- **`polylogue explain "<query>"` as a first-class teaching verb** — the machinery exists (`query_explain.py`, `explain_query_expression` MCP tool, `parser_diagnostics.py`) but is not surfaced as a learn-the-DSL affordance. Wire a top-level `explain` verb that echoes the parsed AST, the selected unit, the retrieval lane, and the lowered SQL in plain English — turning every failed/curious query into a lesson at point of use. — bead/NEW
- **Self-teaching `polylogue tour` (promote demo tour to a guided, narrated walkthrough)** — `demo tour` currently runs and writes artifacts but doesn't *teach*; make an interactive/`--narrate` mode that pauses between stages ("now we search — here's the envelope; now we audit an exit code; now we resume") so the tour is the onboarding doc that can't drift. — NEW
- **`polylogue explain-project` / orientation command** — a cold operator on a fresh checkout has no in-terminal map of the five tiers, the daemon, and the query-first floor. A single command that prints the ring diagram, where the archive lives (XDG paths resolved live), what's ingested, and "your next 3 commands" collapses the README-then-run gap into the tool itself. — NEW
- **Worked-example epilogs on every query verb `--help`** — `find`/`read`/`analyze`/`mark --help` should end with 2-3 real invocations (Click `epilog`), so the reference is learnable without leaving the terminal. Generate these from the same examples the doc-tests execute, keeping `--help`, `cli-reference.md`, and how-tos from one source. — NEW (ties the doc-test idea to point-of-use help)
- **Generated-vs-authored provenance banner + coherence gate across all generated docs** — `docs/README.md` and `cli-reference.md` carry `<!-- Generated by … -->` headers, but coverage is uneven and `render all --check` prints per-surface `sync OK` yet exits 1 (a known trap). Add a `devtools render docs-provenance` audit that asserts *every* generated doc has a machine-readable provenance front-matter block and that no authored doc silently duplicates a generated one. — bead/NEW (extends existing render honesty)
- **A "concept spine" that resolves the Provider/Origin/Source vocabulary confusion for readers** — the docs deliberately carry three overlapping origin vocabularies mid-retirement (`provider-origin-identity.md`, glossary); a cold reader hits `--origin` in one place and `provider` metadata in another and can't tell if it's a bug. Add a single canonical "one concept, three names, here's why, here's which you type" callout transcluded into search.md/data-model.md/glossary so the transitional state reads as intentional, not broken. — bead polylogue-9e5.8 / polylogue-2qx adjacent
- **Doc-linked "cite this" affordance — every claim in a how-to points at the evidence store** — Polylogue's whole pitch is "raw evidence next to derived claim"; the docs should model it. How-to recipes should end with the `raw_artifacts`/`readiness_check`/`explain_import` command that lets the reader verify the doc's own example output, making the "a stranger can cite" bar literal. — NEW
- **A glossary that is executable, not just prose** — `glossary.md` defines material_origin, lineage, tiers in words; add a `polylogue explain-term <name>` (or `--glossary`) that pulls the same definitions from a single YAML the doc renders from, so a term's meaning is one keystroke away in the CLI and can't drift from the doc. — NEW
- **Onboarding "install reality check" — fail the doc when the promised path isn't real** — `getting-started.md` honestly says "no packaged PyPI/Homebrew/container install path is documented as current"; that honesty should be *enforced* by a lint that fails if docs ever mention a `pip install polylogue`/brew/container path before the release artifact exists and smoke-tests green. Keeps onboarding docs from over-promising. — NEW (extends release-readiness-gate)
- **Provider-onboarding matrix: "you use X → here's your exact 3 commands"** — a huge share of first-run friction is "does Polylogue see *my* tool?". `init` scans for roots, but docs don't give per-provider (Claude Code / Codex / ChatGPT-export / Gemini / browser-capture) copy-paste onboarding with the expected `readiness_check` output. Generate this matrix from the same source-detection config `dispatch.py` uses, so a new parser automatically appears in onboarding docs or the render fails. — bead/NEW

GPT-pro prompt stubs:

- **[DR]** "Survey how developer-tool projects (git, DuckDB, ripgrep, sqlite, jq, Datasette) structure the cold-reader path from install → first real result → citing/verifying output. Extract concrete patterns for: task-oriented 'how-to' layers over reference docs, executable/tested documentation examples, and in-CLI self-teaching (`explain`, `tour`, worked-example `--help` epilogs). Produce a ranked pattern catalog with tradeoffs for a local-first, query-first Python CLI whose docs are partly machine-generated."

- **[A]** "Given this docs tree (architecture/internals/daemon reference + a generated docs map + a query-DSL grammar spec + a demo `tour`), design a 'how-do-I' recipe layer and an in-terminal `explain`/`tour`/`explain-project` self-teaching surface. Specify: recipe template, single-source-of-truth so `--help` epilogs, `cli-reference.md`, and how-tos never diverge, and the CI doc-test that executes fenced example blocks against a throwaway demo archive and diffs output. Output file/command names and the render-honesty gate."

- **[A]** "Polylogue's docs carry three overlapping origin vocabularies (Provider/Origin/Source) mid-retirement, and a generated `docs/README.md` map alongside authored subsystem docs. Design (a) a single 'concept spine' callout that makes the transitional vocabulary read as intentional and is transcluded everywhere the terms appear, and (b) an executable glossary (`explain-term`) rendered from one YAML so CLI and docs can't drift. Give the provenance/coherence lint that asserts every generated doc is labeled and no authored doc silently shadows a generated one."

---

## [afca2dca3b42dda68] TIME / TEMPORAL DOCTRINE — 14 concrete ideas (grounded in `polylogue-cpf`, `sort_key_ms`/`

TIME / TEMPORAL DOCTRINE — 14 concrete ideas (grounded in `polylogue-cpf`, `sort_key_ms`/`created_at_ms`/`updated_at_ms`/`acquired_at_ms`/`occurred_at_ms`, `timing_provenance`, `core/dates.py`, `frozen_clock`):

- **Name the four times as a typed enum on every timestamp column** (`source_time` = provider-asserted `created/updated_at_ms`; `ingest_time` = `acquired_at_ms`; `derived_order` = `sort_key_ms`; `event_time` = `occurred_at_ms`) — today they're bare `*_ms` ints indistinguishable at the type level, so a query silently mixes "when the human typed" with "when we noticed the file"; a `TimeKind` newtype makes cross-kind comparison a mypy error — cpf.1 sibling / NEW.

- **`sort_key_ms` must carry its own provenance, not just borrow `timing_provenance`** — `COALESCE(sort_key_ms, created_at_ms, updated_at_ms, 0)` (archive.py:1076) means a session with no source time sorts at epoch-0 and vanishes from `since:` windows invisibly; store `sort_key_provenance` (explicit / inherited-from-message / fell-back-to-ingest / synthesized-0) so "ordered by real time" vs "ordered by acquire order" is queryable, not silent — NEW.

- **Fix the import-time `RELATIVE_BASE` in `core/dates.py`** — `RELATIVE_BASE: datetime.now(...)` is evaluated once at module import, so `since:7d` anchors to *process-start*, not query-time, and drifts in a long-lived daemon; also unpatchable by `frozen_clock`. Make it a call-time param threaded from the request clock — this is a live correctness + test-hermeticity bug — NEW / cpf.

- **Skew-tolerance band as doctrine, not a magic number** — the cpf text says "skew tolerance"; make it real: a documented `CLOCK_SKEW_TOLERANCE_MS` such that two events within the band are treated as *co-temporal* (unordered) rather than falsely ordered by sub-second noise across providers whose clocks differ. Ordering within the band falls back to the stable tiebreak, not to a lie — NEW.

- **Total stable ordering tiebreak spec: `(sort_key_ms, session_id, position, block_position)`** — collisions and NULLs currently rely on ad-hoc `ORDER BY … , s.session_id` in some paths but not all (e.g. cost buckets omit it); codify one canonical comparator so identical-timestamp rows never reorder between two reads of the same corpus — a citation-anchor prerequisite (findings must re-resolve to the same row) — NEW.

- **"When did this REALLY happen" honesty field on read payloads** — every timeline/thread payload should expose `time_confidence` (provider-asserted UTC epoch / provider-asserted local-guessed-UTC / inferred-from-neighbors / ingest-only) so a surface can render "≈ 14:00 (source clock, tz unknown)" instead of a false-precision ISO string. Ties to construct-validity-as-substrate — NEW.

- **Timezone is unknown by default and must be modeled as such** — `parse_date` forces everything to UTC, but Claude web / ChatGPT exports give wall-clock strings whose original tz is often lost; store a nullable `source_tz` and a `tz_provenance` (explicit-offset / assumed-UTC / device-inferred) rather than silently stamping UTC, so "9am my time" survives — NEW.

- **Retroactive/backfilled ingest must never move `sort_key_ms`** — a session re-ingested months later (backfill, GDPR export) has `acquired_at_ms` = now but `source_time` = old; assert an invariant test that a backfilled session lands in its *historical* window, not today's, and that re-ingest with matching content-hash never rewrites `sort_key_ms`. Guards the analytics-one-measure-away lane against phantom "activity spikes" on import days — NEW.

- **Duration honesty ladder tied to `timing_provenance`** — `wall_duration_ms` defaults to `sort_key_estimated` (inter-message gaps), which conflates "human went to lunch" with "model thought hard"; formalize rungs (explicit-provider-window > paired-tool-call-window > first-last-message-span > single-message-zero) and refuse to sum durations across rungs without flagging the weakest rung in the aggregate — extends the degrade-loudly doctrine (cpf.4) — NEW.

- **`since`/`between` semantics must declare which time-kind they filter** — CLI `since:7d` currently hits `sort_key_ms`, but a user asking "sessions since Monday" may mean source-time while a maintenance query means ingest-time; make the field explicit (`ingested:>7d` vs `since:7d`) and document the default, so temporal predicates over skewed sources are unambiguous — cpf / NEW.

- **Half-open interval convention `[since, until)` documented and tested** — `between` boundary inclusivity is currently implicit in `>= ? … <= ?`; pin it (inclusive-start, exclusive-end) so day/month bucket edges (`strftime('%Y-%m', …/1000)`) don't double-count the boundary event in two adjacent buckets — NEW.

- **Monotonic vs wall-clock separation for daemon/ingest metrics** — `frozen_clock` patches `time.time` and `time.monotonic` to the *same* cursor, but production must not measure a convergence-stage *duration* with wall-clock (NTP steps, DST) — audit that all elapsed-time measurements use `monotonic` and all persisted event stamps use wall-UTC-epoch-ms, and add a lint that flags `time.time()` used for a delta — cpf / NEW.

- **Bucketing does UTC-midnight, but "days" are lived in local time** — every `strftime('%Y-%m'/'%Y-%m-%d', ms/1000, 'unixepoch')` rollup silently buckets by UTC calendar day; a 11pm-local session lands on "tomorrow". Offer an operator `report_tz` so calendar/day analytics match the human's lived day, with the UTC-vs-local choice recorded in the finding's provenance stanza — NEW.

- **Temporal model as a first-class doctrine doc + tripwire fixture** — land `docs/doctrine/time.md` (four times, UTC-epoch-ms canon, skew band, duration ladder, tz-unknown default, half-open intervals, backfill invariant) linked from architecture-spine, plus the cpf.1 lint rejecting `TEXT`/ISO-string timestamp columns in new durable DDL and a deny-lexicon fixture rejecting `datetime.now`/naive-tz in new time code — this is the epic's own AC, made concrete for the temporal lane — cpf / cpf.1.

---

GPT-pro prompt stubs:

- **[A]** "Given four coexisting timestamps per event (provider source-time, ingest/acquire-time, derived sort-key, event-time), all stored as UTC epoch-ms across providers with unknown/lossy original timezones and clocks that disagree by seconds-to-minutes: design a total, stable, skew-tolerant ordering with an explicit provenance tag per key, such that (a) identical corpora always yield identical order, (b) sub-second cross-provider noise never forces a false order, and (c) backfilled sessions sort by history not import date. Specify the comparator, the skew band's semantics, and the invariant tests."

- **[DR]** "Survey how mature local-first / event-sourcing / observability systems (SQLite-based archivers, Datomic, OpenTelemetry, Git, email/IMAP, journald) distinguish event-time vs ingest-time vs processing-time, handle clock skew and unknown timezones in retroactively imported records, and represent temporal uncertainty/confidence to users. Extract concrete conventions (half-open intervals, monotonic-vs-wall separation, tz-provenance, late/backfilled data) applicable to a multi-provider AI-chat archive."

- **[DR]** "Research the failure modes of relative date parsing (`since:7d`, `between`) and calendar bucketing (`strftime` UTC-day rollups) in analytics systems where the query-time clock, the data's source clock, and the operator's local timezone all differ — including import-time-frozen relative bases, DST/NTP steps corrupting durations, and UTC-midnight buckets misassigning late-evening local activity. Return best-practice patterns and test strategies (frozen clocks, hermetic time injection)."

---

## [a63f79a463b66530d] THE MCP SURFACE AS A PRODUCT — 14 ideas

Grounded. Key correction: polylogue already ships all three MCP primitives (8 read-only `polylogue://` resources, ~5 hardcoded prompts, ~130 tools, clean hierarchical read/write/admin gating). The real gaps are that resources are static/unsubscribed and prompts are hardcoded rather than derived from the user's own saved recall-packs/views. Ideas build on that reality.

---

**THE MCP SURFACE AS A PRODUCT — 14 ideas**

- Collapse the ~130 tools to a small verb algebra mirroring the CLI's `find→action` shape: one `query` tool taking the Lark DSL expression string (already in `explain_query_expression`/`query_units`) supersedes archive_search/list/aggregate/facets/get_stats_by/session_profiles/threads — the DSL is the composition layer, so ~40 read tools become args, not tools — NEW

- Keep insight tools as a *thin projection over the same `query` verb* (`query "... with cost"` / `... | group by`) instead of 25 parallel `session_costs`/`cost_rollups`/`tool_usage`/`workflow_shape_distribution` endpoints; each insight is a `with <units>` projection already — the MCP duplicates what the DSL unifies — NEW

- Turn the 8 static `polylogue://` resources into the primary READ surface (agents fetch `polylogue://session/{id}` as a resource, not via `get_session` tool) — resources are cacheable, referenceable, and don't burn a tool-call turn; move read-heavy get/list off the tool budget — NEW

- Add `resources/subscribe` on `polylogue://session/{id}` and `polylogue://origin/{name}/recent` so a live agent gets pushed the divergent tail as the daemon ingests it — this is the actual "live session continuity" primitive the CLI can't offer; the daemon already knows when a session's hash changes — NEW

- Add a `resources/list_changed` notification driven off the ingest cursor so an agent watching a workspace learns when new sessions land without polling `list_sessions` — turns the archive into a push feed, the real continuity API — NEW

- Make saved recall-packs and saved-views into **dynamically-registered MCP prompts** (slash-commands): `save_recall_pack` should mint a `/recall-<name>` prompt so the user's curated context becomes a first-class agent affordance, not a tool the agent must remember to call with the right name — bridges `save_recall_pack`/`list_recall_packs` to `server_prompts.py` — NEW

- Replace the 4 hardcoded prompts (analyze_errors/summarize_week/extract_code) with prompts *parameterized by a saved DSL query* — a prompt is "a named query + a rendering instruction," so the prompt catalog should be user-authored via saved-views, not frozen in source — NEW

- Unify the continuity cluster (`get_resume_brief`, `find_resume_candidates`, `find_abandoned_sessions`, `find_stuck_sessions`, `get_postmortem_bundle`, `compose_context_preamble`, `build_context_image`, `compile_context`) into one `continuity` tool with a `mode` enum — these are 8 tools answering one question ("where do I pick up?") with overlapping payloads; an agent can't discover which to call — NEW

- Collapse the mark/tag/annotation/metadata/correction mutation family (~15 write tools: add_mark/remove_mark/add_tag/remove_tag/bulk_tag/save_annotation/delete_annotation/set_metadata/delete_metadata/record_correction/clear_corrections) into one `assert`/`retract` pair over the unified `assertions` table + `AssertionKind` — the storage tier already unified these; the MCP surface didn't follow — matches `user.db` design — NEW

- Fix role-gating coherence: split the axis so *reads that mutate user.db* (save_recall_pack, save_saved_view, save_workspace, blackboard_post) sit at `write`, not scattered — audit that every `save_*`/`delete_*` requires `write` and every `ops reset`/maintenance requires `admin`; today read/write/admin is a clean order but the per-tool assignment needs a contract test enforcing "no user.db write at read role" — Ref TOOL_CONTRACT — NEW

- Make `resolve_ref` + citation-anchor the spine of every payload: every tool result should carry a stable `polylogue://` ref for each session/message/block it names, so an agent can round-trip a finding back into a resource fetch without re-searching — cements "findings-as-objects" from prior waves at the protocol layer — NEW

- Cut the near-duplicate pairs that inflate discoverability cost: `correlate_session`/`correlate_sessions`, `find_similar_sessions`/`neighbor_candidates`, `list_sessions`/`archive_list_sessions`, `search`/`archive_search_sessions`, `get_session`/`archive_get_session`/`get_logical_session` — the `archive_*` and bare aliases are two names for one thing; pick one, delete the other from EXPECTED_TOOL_NAMES — NEW

- Add a single `what_now` router prompt (not tool) that composes readiness + resume-candidates + ready workspace into one MCP prompt an agent invokes at session start — the entry-point should be a *prompt* (user-triggerable slash command) not a tool the agent guesses to call; this is the discoverability anchor for the whole surface — build beyond the activation agent — NEW

- Publish a machine-readable **tool-taxonomy resource** (`polylogue://tools/map`) grouping the surface into read-algebra / continuity / assertions / insights / admin with "prefer X over Y" hints, so an agent doing progressive disclosure loads the map first and never faces 130 flat names — turns EXPECTED_TOOL_NAMES into a navigable graph — NEW

**Cross-cutting thesis:** the ideal shape is ~15-20 tools (query-algebra + continuity + assert/retract + admin), the read surface pushed onto subscribable `polylogue://` resources, and the user's saved recall-packs/views surfaced as dynamic prompts. The MCP server — not the CLI — is the continuity API precisely because only it has resources (referenceable state) and prompts (user-authored affordances); today it under-uses both.

---

**GPT-pro prompt stubs**

- **[A]** "Given a Python MCP server (FastMCP) exposing ~130 tools, 8 static resources, and 5 hardcoded prompts over an AI-session archive with a Lark query DSL that already unifies search/list/aggregate/insight projections (`... | group by | count`, `with <units>`), design the minimal tool set. Show which current tools collapse into a single `query(expression)` verb vs. which need dedicated tools (side-effecting writes, continuity routers), and give the EXPECTED_TOOL_NAMES delta with a migration path that keeps role-gating (read/write/admin) coherent."

- **[DR]** "Research current (2025-2026) best practices for MCP servers that expose all three primitives — tools, resources, and prompts. When should read operations be resources vs. tools? How do production MCP servers use `resources/subscribe` and `list_changed` notifications for live/streaming state? How are user-authored saved queries surfaced as dynamic prompts (slash-commands)? Cite the spec and real server implementations; contrast tool-heavy vs. resource-heavy designs and their agent-ergonomics tradeoffs."

- **[DR]** "For an agent-facing continuity API backed by an append-only session archive with a daemon-driven ingest cursor, research patterns for push-based context delivery over MCP: resource subscriptions vs. polling, notification fan-out, cache-invalidation semantics of `polylogue://session/{id}` when a session's content hash changes, and how citation-anchor URIs let an agent round-trip a search result back into a fetch. Include failure modes (stale subscriptions, unbounded change streams) and mitigations."

---

## [ad6aa83d2423b3843] Grounding confirmed the central tension in this lane: the async `Polylogue` facade (130 me

Grounding confirmed the central tension in this lane: the async `Polylogue` facade (130 methods across 4 mixins) is *not* how the flagship external consumer actually consumes polylogue. Lynchpin reaches in three non-facade ways — `from polylogue.insights.archive import SessionProfileInsightQuery` (substrate-internal import, violates the layering rule), raw `sqlite3.connect(index.db)` fallbacks (a function literally named `_profiles_from_facade` reads the DB directly on "the stable schema", still referencing pre-split `polylogue.db`), and a reimplemented dataclass layer (`lynchpin/sources/polylogue_models.py` duplicating `WorkEvent`/`SessionProfile`). The oracle digest shells out to the CLI. The facade is async-only; every real consumer is synchronous. The `api/contracts/` Protocols exist but only cover a narrow #859 read subset, not the fat surface.

THE PYTHON API / LIBRARY-CONSUMER SURFACE — ideas:

- Sync facade (`Polylogue.sync` / blocking mirror) — the async-only surface is *the* reason lynchpin and oracle bypass it (asyncio.run-per-call is untenable in sync materialization + notebooks); a sync wrapper reclaims those consumers onto the supported path — NEW
- Narrow versioned public SDK namespace (`polylogue.sdk`/`polylogue.public`) with explicit `__all__` + semver — the 130-method facade carries no stability contract, so consumers import `insights.archive` internals; publish the ~20 methods analysts actually need and mark the rest internal — NEW
- One re-export module of frozen public return types (`polylogue.models`) — consumers today either import internal `Session`/`SessionSummary`/`ArchiveInsightModel` or reimplement them (lynchpin's `polylogue_models.py`); a stable typed model namespace ends the duplication — NEW
- Query-as-returnable-object — `pl.query("repo:x since:7d")` yields a reusable `Query` with `.count()/.units()/.to_sql()/.save()` you can pass, cache, and re-run; extends the wave "queries-as-objects" theme into the library ergonomics — NEW
- Columnar bridge (`query.to_arrow()/to_polars()`) — lynchpin's whole substrate is DuckDB and it reads `index.db` raw because there's no columnar handoff; an Arrow return kills the raw-sqlite bypass and makes polylogue a real analytical source — NEW
- Documented stable read-view layer to ATTACH — publish a versioned SQL view surface over `index.db` for columnar consumers instead of the current undocumented direct-table reads against a stale `polylogue.db` name; a contract, not a leak — NEW
- Result objects embed resolvable citation anchors — every returned row/insight carries its `ObjectRef` so a consumer can round-trip back to source/session; builds the citation-anchor theme into the SDK payload shape (pairs with `resolve_ref`) — NEW
- Construct-validity metadata inline on payloads — carry confidence / evidence-count / method-provenance on the returned object rather than behind a separate `insight_rigor_audit` call, so downstream (`ai_session_efficiency` reconstructing "degraded"/confidence semantics) reads it directly — NEW
- Composable measure-algebra API (`measures.cost`, `measures.engaged_minutes`, `.by(day).over(query)`) — lynchpin reimplements engaged-minutes/session aggregation; a first-class measure object over unit sources makes those rollups a one-liner and canonical — NEW
- Typed error taxonomy for consumers (`SessionNotFound`, `SchemaMismatch`, `ArchiveNotReady`) — the facade returns `None` or lets `sqlite3.Error` escape (lynchpin catches it broadly with a timeout probe); named exceptions let consumers branch precisely — NEW
- Schema-binding pin-and-warn at `open()` — SDK records the `index.db`/`user.db` versions it was built against and emits a structured warning on drift; lynchpin reads raw tables blind and would silently misparse after a derived-tier rebuild — NEW
- Capability/introspection call (`pl.capabilities()`) returning the typed surface map + bound schema versions — lets a consumer assert compatibility programmatically instead of duck-typing against a moving 130-method object — NEW
- Canonical composed-Episode return unit — `get()` composes lineage in only some read paths (#2470); the SDK should always return one lineage-composed Episode so consumers never re-derive forks/resumes; anchors the wave "episode unit" theme on the library side — NEW (ref #2467/#2470)
- Streaming/cursor iterators as first-class SDK citizens — `iter_messages` exists but insight readers return full lists; large-archive notebook use needs bounded/backpressured cursors (`.stream()`) so a 38GB archive doesn't materialize in RAM — NEW
- Example-driven, CI-doctested SDK cookbook (`docs/sdk-cookbook.md`) — there are zero runnable library-consumer examples today; snippets executed in CI double as the DX doc and a drift guard on the public surface — NEW

GPT-pro prompt stubs:

- [A] "Design a minimal versioned public Python SDK (`polylogue.sdk`) extracted from a 130-method async god-facade whose only real external consumer bypasses it via substrate-internal imports, raw SQLite reads, and reimplemented models. Specify: the ~20-method surface, sync-vs-async story, frozen return-type namespace, error taxonomy, and a deprecation path for the fat facade — with the layering constraint that surfaces may not import substrate internals."
- [A] "Design a columnar/DataFrame consumer contract for an append-mostly SQLite analytical store (rebuildable `index.db`) feeding a downstream DuckDB hub. Compare: (a) stable versioned SQL views to ATTACH, (b) Arrow/Polars return objects from a query builder, (c) a materialized export tier. Address schema-version binding, drift detection, and zero-copy vs isolation tradeoffs."
- [DR] "Survey how mature local-first / embedded analytical Python libraries (DuckDB, ibis, Polars, datasette, LanceDB, sqlite-utils) expose a stable library-consumer surface: query-object ergonomics, sync/async ergonomics for notebooks, typed return payloads, versioning of the public API vs internals, and example-driven docs. Extract concrete patterns applicable to turning a session-archive facade into an SDK a data analyst actually wants."